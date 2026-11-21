// Headless settings facade. Defaults copied from Tartini 1.2 global/gdata.cpp.
#pragma once
#include <algorithm>
#include "useful.h"
enum AnalysisModes { MPM, AUTOCORRELATION, MPM_MODIFIED_CEPSTRUM };
struct GData {
    double thresholds[7][2] = {{-85,0},{-30,-20},{.4,1},{.5,.02},{0,.1},{.03,.2},{.12,.3}};
    int mode = MPM;
    int analysisType() { return mode; }
    bool doingFreqAnalysis() { return true; }
    bool doingActiveAnalysis() { return true; }
    bool doingAutoNoiseFloor() { return true; }
    bool doingHarmonicAnalysis() { return true; }
    double dBFloor() { return -150; }
    double topPitch() { return 128; }
    double ampThreshold(int i, int j) { return thresholds[i][j]; }
    double &rmsFloor() { return thresholds[0][0]; }
    double &rmsCeiling() { return thresholds[0][1]; }
};
extern thread_local GData *gdata;
