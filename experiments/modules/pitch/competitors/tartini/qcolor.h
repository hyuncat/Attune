// Colors never participate in analysis.
#pragma once
namespace Qt { const int black = 0; }
struct QColor { QColor(int = 0) {} };
