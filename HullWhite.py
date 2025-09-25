class OneFactorHullWhiteModel:
    def __init__(self, a : float, sigma : float) -> None:
        if sigma <= 0:
            raise Exception("sigma must be positive.")
        self.a = a
        self.sigma = sigma