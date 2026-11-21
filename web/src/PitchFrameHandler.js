// Hand-written refactor of looping over pitch frames for web environment

const UNVOICED_THRESHOLD = 0.9;

export function getVoicedPitchFrames(pitchFrames, t0, t1) {
    const voicedFrames = [];
    for (const frame of pitchFrames ?? []) {
        if (!frame) continue;
        const [time, candidates, volume, unvoicedProb, , alignedDistance, isTransition, value] = frame;
        if (time < t0 || time > t1) continue;
        const voiced = value != null && value !== -1 && unvoicedProb < UNVOICED_THRESHOLD;
        if (voiced){
            voicedFrames.push(frame);
        }
    }
    return voicedFrames;

}

// No voicing filter at all, unlike getVoicedPitchFrames above - some callers
// (e.g. volume curves) need unvoiced/noisy frames too, since loudness is
// still meaningful for them. Deliberately its own function rather than a
// boolean flag on getVoicedPitchFrames: that flag could get flipped by
// accident later and silently start dropping frames a caller depends on.
export function getPitchFramesInRange(pitchFrames, t0, t1) {
    const framesInRange = [];
    for (const frame of pitchFrames ?? []) {
        if (!frame) continue;
        const [time, candidates, volume, unvoicedProb, , alignedDistance, isTransition, value] = frame;
        if (time < t0 || time > t1) continue;
        framesInRange.push(frame);
    }
    return framesInRange;

}


