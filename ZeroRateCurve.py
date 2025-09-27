from abc import ABC, abstractmethod

class ZeroRateCurve(ABC):
    @abstractmethod
    def get_zero_rate(self, t: float) -> float:
        """
        Return the zero rate for maturity t (in years).
        """
        pass

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
