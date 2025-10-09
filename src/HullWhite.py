import bisect

class OneFactorHullWhiteModel:
    def __init__(self, a: float) -> None:
        self.a = a
        self.sigma_breakpoints = [0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 30]
        self.sigma_values = [0.02, 0.02, 0.02, 0.015, 0.015, 0.015, 0.012, 0.01, 0.01, 0.01, 0.01, 0.01]

    def set_constant_sigma(self, sigma):
        self.sigma_values = [sigma] * len(self.sigma_values)

    def set_sigmas_from_vector(self, vector):
        if len(vector) != len(self.sigma_values):
            raise Exception(f"Expected {len(self.sigma_values)} sigma values, got {len(vector)}")
        self.sigma_values = vector.copy()

    def sigma(self, t):
        if t < 0:
            raise Exception("t can't be negative!")

        # below first breakpoint
        if t <= self.sigma_breakpoints[0]:
            return self.sigma_values[0]
        
        # above last breakpoint
        if t >= self.sigma_breakpoints[-1]:
            return self.sigma_values[-1]

        # find interval
        idx = bisect.bisect_left(self.sigma_breakpoints, t)
        t0, t1 = self.sigma_breakpoints[idx - 1], self.sigma_breakpoints[idx]
        sigma0, sigma1 = self.sigma_values[idx - 1], self.sigma_values[idx]

        # linear interpolation
        sigma_t = sigma0 + (sigma1 - sigma0) * (t - t0) / (t1 - t0)
        return sigma_t
