// Headless C ABI for the original Tartini 1.2 analysis engine.
#include "channel.h"
#include <cmath>
#include <memory>
thread_local GData *gdata = nullptr;
struct Engine {
    GData settings;
    SoundFile file;
    std::unique_ptr<Channel> channel;
    Engine() {
        gdata = &settings;
        file.myTransforms.init(2048, 0, 44100, true);
        channel.reset(new Channel(&file, 2048));
    }
};
extern "C" {
void *tartini_create() { return new Engine(); }
void tartini_destroy(void *handle) {
    auto *e = static_cast<Engine *>(handle);
    gdata = &e->settings;
    delete e;
    gdata = nullptr;
}
// Append a hop (first frame contains half a window of implicit leading zeros).
int tartini_push(void *handle, const float *samples) {
    auto &e = *static_cast<Engine *>(handle);
    gdata = &e.settings;
    auto &ch = *e.channel;
    ChannelLocker lock(&ch);
    ch.shift_left(1024);
    std::copy(samples, samples + 1024, ch.directInput.end() - 1024);
    ch.highPassFilter->filter(samples, ch.filteredInput.end() - 1024, 1024);
    for (int j = 1024; j < 2048; ++j)
        ch.filteredInput[j] = bound(ch.filteredInput[j], -1.0f, 1.0f);
    FilterState state;
    static_cast<IIR_Filter *>(ch.highPassFilter)->getState(&state);
    ch.processNewChunk(&state);
    // SoundFile::readChunk/finishRecordChunk advance the cursor AFTER analysis.
    e.file.chunk = ch.totalChunks() - 1;
    return ch.totalChunks();
}
// Mode 0 = stored fundamental frequency; 1 = visible pitch contour.
float tartini_frequency(void *handle, int chunk, int visible) {
    auto &e = *static_cast<Engine *>(handle);
    gdata = &e.settings;
    auto *data = e.channel->dataAtChunk(chunk);
    if (!data || (visible && !e.channel->isVisibleChunk(data))) return 0;
    return std::isfinite(data->fundamentalFreq) ? data->fundamentalFreq : 0;
}
int tartini_note(void *handle, int chunk) {
    auto &e = *static_cast<Engine *>(handle);
    return e.channel->dataAtChunk(chunk)->noteIndex;
}
void tartini_nsdf(const float *samples, float *output) {
    GData settings;
    gdata = &settings;
    MyTransforms transforms;
    transforms.init(2048, 0, 44100, false);
    transforms.nsdf(const_cast<float *>(samples), output);
    gdata = nullptr;
}
}
