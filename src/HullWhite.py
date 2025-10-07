class OneFactorHullWhiteModel:
    def __init__(self, a: float) -> None:
        self.a = a
        self.sigma_values = [
            0.01,  # 1y
            0.01,  # 2y
            0.01,  # 3y
            0.01,  # 4y
            0.01,  # 5y
            0.01,  # 7y
            0.01,  # 10y
            0.01,  # 12y
            0.01,  # 15y
            0.01,  # 20y
            0.01,  # 25y
            0.01   # 30y
        ]
        self.sigma_breakpoints = [2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30]

    def set_constant_sigma(self, sigma):
        self.sigma_values = [sigma] * len(self.sigma_values)

    def set_sigmas_from_vector(self, vector):
        if len(vector) != len(self.sigma_values):
            raise Exception(f"Expected 12 sigma values, got {len(vector)}")
        self.sigma_values = vector.copy()

    def sigma(self, t):
        if t < 0: raise Exception("t can't be negative!")

        for i, t_upper in enumerate(self.sigma_breakpoints):
            if t < t_upper:
                return self.sigma_values[i]
        return self.sigma_values[-1]
