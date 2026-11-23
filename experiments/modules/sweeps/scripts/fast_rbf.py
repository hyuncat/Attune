"""Same ruptures RBF kernel, with constant-time segment sums."""
import numpy as np
from ruptures.costs import CostRbf
from ruptures.exceptions import NotEnoughPoints


class PrefixCostRbf(CostRbf):
    def fit(self, signal):
        # Preserve ruptures' median gamma, clipping, and diagonal convention.
        self._gram = None
        super().fit(signal)
        gram = self.gram
        self._diagonal = np.r_[0., np.cumsum(np.diag(gram))]
        # Reuse the existing quadratic allocation rather than keeping two matrices.
        np.cumsum(gram, axis=0, out=gram)
        np.cumsum(gram, axis=1, out=gram)
        self._prefix = gram
        return self

    def error(self, start, end):
        if end - start < self.min_size:
            raise NotEnoughPoints
        p = self._prefix
        total = p[end - 1, end - 1]
        if start:
            total = total - p[start - 1, end - 1] - p[end - 1, start - 1] + p[start - 1, start - 1]
        return self._diagonal[end] - self._diagonal[start] - total / (end - start)
