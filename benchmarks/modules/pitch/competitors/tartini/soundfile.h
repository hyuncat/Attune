// Transport facade; DSP and Channel/NoteData algorithms come from the archive.
#pragma once
#include "gdata.h"
#include "mytransforms.h"
#include <mutex>
struct QMutex {
    std::recursive_mutex mutex;
    explicit QMutex(bool) {}
    void lock() { mutex.lock(); }
    void unlock() { mutex.unlock(); }
};
class QString;
class SoundFile {
public:
    MyTransforms myTransforms;
    bool firstTimeThrough = true;
    int chunk = 0;
    int rate() { return 44100; }
    int framesPerChunk() { return 1024; }
    double timePerChunk() { return 1024.0 / rate(); }
    double startTime() { return 0; }
    void setStartTime(double) {}
    void jumpToTime(double) {}
    int chunkAtTime(double t) { return int(t / timePerChunk()); }
    double chunkFractionAtTime(double t) { return t / timePerChunk(); }
    int chunkAtCurrentTime() { return chunk; }
    int currentChunk() { return chunk; }
    int currentRawChunk() { return chunk; }
    int currentStreamChunk() { return chunk; }
    double timeAtChunk(int c) { return c * timePerChunk(); }
    bool doingDetailedPitch() { return false; }
};
