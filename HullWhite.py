class ForwardCurve:
    pass

class ThetaProcess:
    pass

class OneFactorHullWhiteModel:
    def __init__(self, a : float, sigma : float, theta : ThetaProcess=None) -> None:
        self.a = a
        self.sigma = sigma
        self.theta = theta

    # def df(self, t, f_t, dt, dz):
    #     return ( (self.theta.get_value(t)-self.alpha*f_t)*dt + self.sigma*dz )

    # def dg(self, t, g_t, dt):
    #     return ( (self.theta.get_value(t) - self.alpha*g_t)*dt )
    
    # def dx(self, t, f_t, g_t, dt, dz):
    #     return self.df(t, f_t, dt, dz) - self.dg(t, g_t, dt)