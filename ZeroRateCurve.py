from abc import ABC, abstractmethod

class ZeroRateCurve(ABC):
    @abstractmethod
    def get_zero_rate(self, t: float) -> float:
        """
        Return the zero rate for maturity t (in years).
        """
        pass

class StepwiseZeroRateCurve(ZeroRateCurve):
    def __init__(self):
        self.curve = [
            [0.25, 0.02046],
            [0.5, 0.02082],
            [1, 0.02085],
            [2, 0.02012],
            [3, 0.02082],
            [4, 0.02205],
            [5, 0.02311],
            [6, 0.02362],
            [7, 0.02473],
            [8, 0.02583],
            [9, 0.02671],
            [10, 0.02758],
            [15, 0.03114],
            [20, 0.03249],
            [30, 0.03379],
        ]
        self.ranges = []
        self._create_ranges()
    
    def _create_ranges(self):
        for i in range(0, len(self.curve)):
            if i == 0:
                self.ranges.append( ([0, self.curve[0][0]], self.curve[0][1]) )
            else:
                self.ranges.append( ([self.curve[i-1][0], self.curve[i][0]], self.curve[i-1][1]) )
    
    def get_zero_rate(self, t):
        for cur_range, cur_rate in self.ranges:
            if cur_range[0] <= t and cur_range[1] >= t:
                return cur_rate
        return self.curve[-1][1]
    


class NielsonSiegelSvenssonCurve(ZeroRateCurve):
    ...

class LinearZeroRateCurve(ZeroRateCurve):
    def __init__(self, r0: float = 0, r30: float = 0.05, max_t: float = 30):
        """
        Simple linear zero rate curve.

        Args:
            r0: Zero rate at t=0 (e.g., 1%).
            r30: Zero rate at t=max_t (e.g., 5%).
            max_t: Maximum maturity (default 30 years).
        """
        self.r0 = r0
        self.r30 = r30
        self.max_t = max_t

    def get_zero_rate(self, t: float) -> float:
        """
        Returns the zero rate for maturity t, using linear interpolation
        between r0 and r30. Extrapolates flat beyond max_t.
        """
        if t <= 0:
            return self.r0
        elif t >= self.max_t:
            return self.r30
        else:
            slope = (self.r30 - self.r0) / self.max_t
            return self.r0 + slope * t
