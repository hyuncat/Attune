import numpy as np
from scipy.signal import find_peaks, iirfilter, sosfilt
from scipy.stats import beta
import threading
from PyQt6.QtCore import QObject, pyqtSignal
from tqdm import tqdm

from app_logic.user.ds.PitchData import Pitch
from app_logic.user.ds.Recording import Recording
from algorithms.Config import Config

class PitchDetector(QObject):
    
    pitch_detected = pyqtSignal(float)
    
    def __init__(self, recording: Recording=None, config: Config=None, parent: QObject|None=None):
        """
        Initialize the pitch detection parameters, like the tuning, frequency range, etc.
        Best to make it as specific as possible to your desired use case to improve accuracy of the detection.
        """
        super().__init__(parent)
        if not recording and not config:
            raise ValueError("Must provide either a recording or a config to initialize the PitchDetector.")
        self.recording = recording
        self.config = config if config else recording.config
        self.SR = self.config.sr # for sample-to-frequency conversion

        # --- pitch config variables ---
        # ensure max lag is big enough to detect lowest f0 (largest period)
        # defaults to violin min
        self.tau_max = int(self.config.sr / self.config.fmin) 
        self.tau_min = int(self.config.sr / self.config.fmax)

        # initialize beta distribution parameters
        self.UNVOICED_PROB = 0.01
        self.N_THRESHOLDS = 100
        self.beta_pdf, self.thresholds = self.threshold_prior(n_thresholds=self.N_THRESHOLDS)

        # rolling window variables (for detect_pitches)
        self.FRAME_SIZE = self.config.w1
        self.HOP_SIZE = self.config.h1

        # threading variables
        self.pda_thread: threading.Thread = None
        self.stop_event = threading.Event()

        # block variable for stalling buffer
        self.block = False

    # might not be necessary anymore :)
    def load_config(self, config: Config):
        """re-initialize the tuning parameters"""
        self.config = config
        self.SR = config.sr
        self.tau_max = int(self.SR / config.fmin) 
        self.tau_min = int(self.SR / config.fmax)

        # rolling window variables (for detect_pitches)
        self.FRAME_SIZE = config.w1
        self.HOP_SIZE = config.h1

    def run(self, start_time: float=None):
        """keep trying to detect pitches while we can"""
        self.stop()
        self.stop_event.clear()
        self.recording.a2p_queue.init_start_time(start_time)
        self.pda_thread = threading.Thread(
            target=self._run, daemon=True
        )
        self.pda_thread.start()
    
    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                # if self.block:
                    # self.recording.a2p_queue.stall(self.HOP_SIZE) # occurs when in practice mode and you fuck up
                x, t = self.recording.a2p_queue.pop(self.FRAME_SIZE, self.HOP_SIZE, stall=self.block)

                if x is None: # returns none if not enough to detect
                    self.stop_event.wait(0.002)
                    continue
                # x, t = x[0], x[1]
                pitch = self.detect_pitch(x, t)
                self.recording.write_pitch_data([pitch], t)
                # print(f'detected pitch @ {pitch.time}, midi_num: {pitch.candidates[0][0]}, unvoiced_prob: {pitch.unvoiced_prob}')
                self.pitch_detected.emit(pitch.time)

            except Exception as e:
                print(f"[PitchDetector] frame skipped due to error: {e}")
                continue

    def stop(self):
        if self.pda_thread and self.pda_thread.is_alive():
            self.stop_event.set()
            self.pda_thread.join() # pause the main thread until recording thread recognizes the stop event

    
    # THE DETECTION ALGORITHM
    def detect_pitch(self, x: np.ndarray, start_time: float=None) -> Pitch:
        """a method to call pitch detection on a single frame
        requires an explicit reference to the start time

        Args:
            x: the array of audio to perform pitch detection on
            start_time: for when we run on longer audio and keeping track of the frame
        """
        # if frame is empty, return unvoiced pitch
        if np.all(x == 0):
            return Pitch(time=start_time, candidates=[], 
                         volume=0.0, unvoiced_prob=1.0, 
                         distance=None, config=self.config)
        
        # preprocess audio to center and get rid of low frequency noise
        x, volume = self.preprocess_audio(x)

        # compute autocorrelation and modify it to avoid 0-lag peak
        acf, _ = self.autocorrelation_fft(x)
        cdf = self.cmndf(x, acf)
        # cdf = self.clamped_diff_fct(x=x, acf=acf)

        # prominence picking + probability assignment to all freq estimates
        acf_peaks = self.find_acf_peaks(acf)
        pitch_probs, unvoiced_prob = self.pitch_probabilities(acf_peaks, cdf)

        # interpolate + compute final freq estimates
        freq_estimates = [self.SR/self.parabolic_interpolation(acf, t) for t in acf_peaks]
        midi_estimates = [self.config.freq_to_midi(f) for f in freq_estimates]

        # create + return the final pitch object
        candidates = list(zip(midi_estimates, pitch_probs))
        candidates.sort(key=lambda c: c[1], reverse=True) # sort from most to least probable
        _note = self.recording.score_data.current_note() if self.recording.score_data else None
        distance = _note.midi_num[0] - candidates[0][0] if _note else None
        # print(f"detected pitch @ {start_time:.2f} sec, midi_num: {candidates[0][0]:.2f}, unvoiced_prob: {unvoiced_prob:.2f}, distance to target: {distance:.2f}")
        pitch = Pitch(time=start_time, candidates=candidates, 
                      volume=volume, unvoiced_prob=unvoiced_prob, 
                      distance=distance, config=self.config)
        return pitch


    def detect_pitches(self, x: np.ndarray) -> list[Pitch]:
        """
        Computes multi-frame pitch detection on an arbitrary length array of audio data.
        Returns a nested list of pitches, each corresponding to the freq estimates (probabilistic)
        for each timestep
        """
        # get memory efficient frames with np pointer c++ magic
        frames = np.lib.stride_tricks.sliding_window_view(x, self.FRAME_SIZE)[::self.HOP_SIZE]
        n_frames = 1 + (len(x) - self.FRAME_SIZE) // self.HOP_SIZE 

        pitches = []

        for i, frame in tqdm(enumerate(frames), total=n_frames):

            start_time = (i*self.HOP_SIZE)/self.SR # elapsed time of the frame
            pitch = self.detect_pitch(frame, start_time)
            pitches.append(pitch)
            
        return pitches



    # METHODS TO IMPLEMENT THE ALGORITHM
    # ---
    def re_init(self, config: Config=None):
        """re-initialize the tuning parameters"""
        if config:
            self.config = config
            self.SR = config.sr
            self.tau_max = int(self.SR / config.fmin) 
            self.tau_min = int(self.SR / config.fmax)

    # the probability distribution of thresholds
    def threshold_prior(self, n_thresholds: int=100, a: float=2, b: float=34/3) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a beta distribution modeling the pdf for YIN thresholds,
        represented as a numpy array of size N_THRESHOLDS corresponding to the pdf values.
        Also returns the array of thresholds.

        Possible a,b parameters from paper:
            - mean=0.1 beta(a=2, b=18)
            - mean=0.15 beta(a=2, b=11.33)
            - mean=0.2 beta(a=2, b=8)

        Not yet sure how these parameters are determined but I got them from the paper.
        """
        # all thresholds
        thresholds = np.linspace(0, 1, n_thresholds+1)
        thresholds = thresholds[1:] # remove the 0 threshold
        beta_pdf = beta.pdf(thresholds, a, b) / n_thresholds
        return beta_pdf, thresholds

    # --- frequency parsing functions ---
    # autocorrelation (base)
    def autocorrelation_fft(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast autocorrelation function implementation using Wiener-Khinchin theorem,
        which computes autocorrelation as the inverse FFT of the signal's power spectrum.

        Step 1 of Yin algorithm, corresponding to equation (1) in Cheveigne, Kawahara 2002.

        Args:
            x: The current frame of audio samples in Yin algorithm
            tau_max: Check for all time lags up to this value for in autocorrelation

        Returns:
            autocorrelation: The similarity curve.
            amplitudes: Amplitudes of the frame.
        """
        x = np.array(x, dtype=np.float64)
        w = x.size
        tau_max = min(self.tau_max, w)

        # zero-pad the audio signal array by the minimum power of 2 which
        # is larger than the window size + tau_max
        min_fft_size = w + tau_max  # (pad by >tau_max for frame end)

        p2 = (min_fft_size // 32).bit_length()
        nice_fft_sizes = (16, 18, 20, 24, 25, 27, 30, 32)
        size_pad = min(size * (2 ** p2) for size in nice_fft_sizes if size * 2 ** p2 >= min_fft_size)

        # --- AUTOCORRELATION WITH WIENER-KHINCHIN ---
        # decompose the signal into its frequency components
        fft_x = np.fft.rfft(x, size_pad)  
        psd = fft_x * fft_x.conjugate() # power spectrum density
        autocorrelation = np.fft.irfft(psd)[:tau_max] 

        amplitudes = np.abs(fft_x)

        # only return valid overlapping values up to window_size-tau_max
        return autocorrelation, amplitudes

    def cmndf(self, x, acf) -> np.ndarray:
        """
        cumulative normalized difference function (CMNDF) for better peak-picking of the fundamental period
        based directly off original YIN algorithm

        Args:
            x: needed to compute energy for diff_fct inversion
            acf: the result of autocorrelation on x
        """
        # --- INVERT TO DIFFERENCE FUNCTION ---
        # compute the energy (r_t(0) and r_{t+\tau}(0)) for each lag
        r_0 = np.sum(x**2)
        energy = np.full(acf.shape, r_0)

        diff_fct = energy[0] + energy - 2*acf
        diff_fct[0] = 0
        diff_fct = np.abs(diff_fct)

        # --- NORMALIZE + CLAMP
        diff_fct = diff_fct / (np.max(diff_fct) - np.min(diff_fct))

        cmndf = np.zeros(self.tau_max) 
        cmndf[0] = 1 # make first value 1
        total_diff = 1

        for tau in range(1, self.tau_max):
            total_diff += diff_fct[tau]
            avg_diff = total_diff / tau 
            cmndf[tau] = diff_fct[tau] / avg_diff

        return cmndf

    # modifying the difference function
    def clamped_diff_fct(self, x, acf) -> np.ndarray:
        """
        modifies the base autocorrelation by inverting + normalizing it, then
        clamping all values outside of desired tau_range to be 1

        Args:
            x: needed to compute energy for diff_fct inversion
            acf: the result of autocorrelation on x
        """
        # --- INVERT TO DIFFERENCE FUNCTION ---
        # compute the energy (r_t(0) and r_{t+\tau}(0)) for each lag
        r_0 = np.sum(x**2)
        energy = np.full(acf.shape, r_0)

        diff_fct = energy[0] + energy - 2*acf
        diff_fct[0] = 0
        diff_fct = np.abs(diff_fct)

        # --- NORMALIZE + CLAMP
        diff_fct = diff_fct / (np.max(diff_fct) - np.min(diff_fct))

        clamp_df = np.zeros(self.tau_max) 
        clamp_df[:self.tau_min] = 1 # make everything before min f_0 1
        total_diff = self.tau_min

        for tau in range(self.tau_min, self.tau_max):
            total_diff += diff_fct[tau]
            avg_diff = total_diff / tau 
            clamp_df[tau] = diff_fct[tau] / avg_diff

        return clamp_df
    
    # --- peak-picking ---
    # prominence-based initial peak-finding
    def find_acf_peaks(self, acf: np.ndarray):
        """prominence-based peak picking of the autocorrelation curve
        returns the indices of all possible tau (fundamental period) values
        """
        # initial prominence as 1/2 overall acf range
        prominence = abs((np.max(acf) - np.min(acf))/2)

        n = 5 # how many times to find peaks within the intial prominence range
        for i in range(0, n):
            # try the lowest prominence we can that still returns valid
            p = prominence - prominence*(i/n) 
            acf_peaks, _ = find_peaks(acf, prominence=p)
            if len(acf_peaks) > 0:
                break

        # fallback if still empty
        if acf_peaks.size == 0:
            # look for the global ACF max in [tau_min, tau_max)
            region = acf[self.tau_min : self.tau_max]
            best = np.argmax(region) + self.tau_min
            acf_peaks = np.array([best], dtype=int)

        return acf_peaks

    # find the pitch according to YIN thresholding
    def find_pitch(self, cdf: np.ndarray, acf_peaks: np.ndarray, threshold: float=0.1) -> tuple[int, bool]:
        """
        Finds the YIN pitch estimate with their absolute thresholding step by searching for the first cdf trough
        below a certain 'absolute threshold'. Runs 

        Args:
            cdf: the clamped difference function for y-values corresponding in the threshold range
            acf_peaks: indices of the prominent-peaks found from the ACF (to index into the CDF)
            threshold: take the first trough below this value of d'(tau)
        
        Returns:
            tau_0 (int): The fundamental period estimate. If possible, the first tau st. 
                         d'(tau) < threshold. Else, the x of the global minima
            tau_idx (int): index of the chosen peak in the acf_peaks array
            is_voiced (bool): False if we return the global min
        """
        for i, min in np.ndenumerate(acf_peaks):
            if cdf[min] <= threshold:
                return min, i[0], True
        
        # no min found below threshold, return the global minima
        i = np.argmin(cdf[acf_peaks])
        global_min = acf_peaks[i]
        return global_min, i, False
    
    # assign remaining peaks a probability with find_pitch
    def pitch_probabilities(self, acf_peaks: np.ndarray, cdf: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Given all prominent-enough peaks of the original ACF curve, 
        computes the probability of all possible period (tau) = 1/f_0 estimates.
        Based off the PYin method of probability assignment.

        Args:
            acf_peaks: indices of all prominent-enough peaks of the original ACF curve
            cdf: clamped difference function, to help for threshold-based peak-picking

        Returns:
            pitch_probs: array of same shape as acf_peaks, where corresponding
                         indices represent associated probabilities
            unvoiced_prob: 1 - sum(tau_probs), eg adding up all the times we had to 
                           take the global min because nothing was below the threshold
        """
        pitch_probs = np.zeros_like(acf_peaks, dtype=np.float64)

        for i, threshold in enumerate(self.thresholds):
            tau_0, j, is_voiced = self.find_pitch(cdf, acf_peaks, threshold)
            # if returned pitch for a threshold is not within pitch range, call it unvoiced
            # (trying to minimize harmonic errors)
            if is_voiced and tau_0 <= self.tau_max and tau_0 >= self.tau_min:
                pitch_probs[j] += self.beta_pdf[i]
            else:
                pitch_probs[j] += self.beta_pdf[i] * self.UNVOICED_PROB
            
        unvoiced_prob = 1 - np.sum(pitch_probs)
        return pitch_probs, unvoiced_prob

    # refine final peak estimates
    def parabolic_interpolation(self, acf: np.ndarray, acf_peak: int) -> float:
        """
        Refines the peak estimates by performing parabolic interpolation around the given index
        of the AACF. Fits a negative quadratic to the supplied minima.
        
        Args:
            acf: A 1D array of y-values (e.g., diff_fct values).
            acf_peak: The index of the maximum point in acf to interpolate around.

        Returns:
            The interpolated x-pos of the supplied acf_peak
        """
        x = acf_peak

        # don't interpolate at boundaries - need at least 3 points
        if x <= 0 or x >= len(acf) - 1:
            return float(x)

        y_1 = acf[x - 1]
        y_2 = acf[x]
        y_3 = acf[x + 1]

        denominator = 2 * (y_1 - 2*y_2 + y_3)
        if denominator == 0:
            return float(x)
        
        x_interpolated = x + (y_1 - y_3) / denominator
        return x_interpolated


    # AUDIO PREPROCESSING
    def bandpass_filter(self, x: np.ndarray, fmin: float=50, fmax: float=4000) -> np.ndarray:
        """
        a 2nd order Butterworth bandpass IIR (infinite impulse response) filter to get 
        rid of noise outside of the typical pitch range

        Args:
            x: The input audio signal as a 1D NumPy array.
            fmin: The lower cutoff frequency (in Hz) for the bandpass filter. Frequencies below this value will be attenuated.
            fmax: The upper cutoff frequency (in Hz) for the bandpass filter. Frequencies above this value will be attenuated.

        Returns:
            np.ndarray: The filtered audio signal as a 1D NumPy array, with reduced noise outside the typical pitch range.
        """
        sos = iirfilter(
            N=2, Wn=[fmin, fmax],
            btype='bandpass', 
            ftype='butter', 
            output='sos', 
            fs=self.SR
        )
        x = sosfilt(sos, x)
        return x
    
    def high_pass_iir_filter(self, x: np.ndarray, cutoff_freq=150, 
                             sr: int=44100) -> np.ndarray:
        """
        a 2nd order high pass IIR (infinite impulse response) filter to lower intensity 
        of low frequency noise below 150 Hz

        Args:
        audio_data (np.ndarray): The input audio signal as a 1D NumPy array.
        cutoff_freq (float, optional): The cutoff frequency (in Hz) for the high-pass filter. 
                                        Frequencies below this value will be attenuated. 
                                        Defaults to 150 Hz.
        sr (int, optional): The sampling rate (in Hz) of the audio signal. 
                            This is used to calculate the Nyquist frequency 
                            for normalizing the cutoff frequency. Defaults 
                            to the sample rate defined in `AppConfig.SAMPLE_RATE`.

        Returns:
            np.ndarray: The filtered audio signal as a 1D NumPy array, with reduced 
                        low-frequency noise.
        """
        nyquist_freq = sr / 2

        # normalize freq by nyquist (scipy expects Wn between 0 - 1)
        normal_cutoff = cutoff_freq / nyquist_freq
        sos = iirfilter(
            N=2, Wn=normal_cutoff, rp=3, 
            btype='highpass', 
            ftype='butter', 
            output='sos', 
            fs=sr
        )
        x = sosfilt(sos, x)
        return x

    def preprocess_audio(self, x: list) -> tuple[np.ndarray, float]:
        """
        centers the audio around mean, normalizes, 
        and applies high pass iir filter to prepare for pitch detection

        Args:
            x (list): The input audio signal as a list of samples.
            iir_cutoff_freq (float, optional): The cutoff frequency for the high-pass filter. Defaults to 150 Hz.

        Returns:
            tuple: A tuple containing the preprocessed audio signal (as a NumPy array) and the volume (as a float).
        """
        if len(x) == 0:
            return np.array([]), 0.0
        x = np.asarray(x, dtype=float)
        # x = x.astype(float)
        x = x - np.mean(x) # center
        volume = np.sqrt(np.mean(x ** 2))  # get volume as mean |amplitude| of the x (before normalizing)
        x = x/np.max(np.abs(x)) # normalize
        x = self.bandpass_filter(x, fmin=self.config.fmin*0.8, fmax=self.config.fmax*1.2) # get rid of noise outside of pitch range
        # x = self.high_pass_iir_filter(x, iir_cutoff_freq)
        return x, volume