from enum import Enum
from src.ZeroRateCurve import ZeroRateCurve
from src.HullWhiteTreeUtil import round_list_floats
import numpy as np

class SwaptionType(Enum):
    PAYER = "payer"
    RECEIVER = "receiver"

class EuropeanSwaption:
    def __init__(self, swaption_type: SwaptionType,
                 expiry: float, swap_start: float, swap_end: float,
                 payment_frequency: float, notional: float, strike: float, 
                 fixed: float) -> None:
        self.swaption_type = swaption_type
        self.expiry = expiry
        self.swap_start = swap_start
        self.swap_end = swap_end
        self.payment_frequency = payment_frequency
        self.notional = notional
        self.strike = strike
        self.fixed = fixed
    
    def set_ATM_strike_fixed_rate_and_strike(self, zcb_curve: ZeroRateCurve):
        self.strike = self.get_par_rate(zcb_curve)
        self.fixed = self.strike
    
    @round_list_floats
    def get_fixed_leg_payment_times(self, fixed_tau=None) -> list[float]:
        # fixed leg is paid annually (ASSUMPTION!!... based on Bloomberg DESC)
        times = []
        cur_time = self.swap_start
        while cur_time < self.swap_end:
            cur_time += 1.0 if fixed_tau is None else fixed_tau
            times.append(cur_time)
        return times

    @round_list_floats
    def get_floating_leg_payment_times(self) -> list[float]:
        return self.get_fixed_leg_payment_times(fixed_tau=self.payment_frequency)

    @round_list_floats
    def get_valuation_times(self) -> list[float]:
        valuation_times = [0.0, self.swap_start]
        cur_time = self.swap_start
        while cur_time < self.swap_end:
            cur_time += self.payment_frequency
            valuation_times.append(cur_time)
        return valuation_times

    def get_par_rate(self, zcb_curve: ZeroRateCurve):
        # calculates the par rate using the par rate formula........
        par_rate_numerator = ( np.exp(-zcb_curve.get_zero_rate(t=self.swap_start)*self.swap_start)
                              - np.exp(-zcb_curve.get_zero_rate(t=self.swap_end)*self.swap_end))
        par_rate_denominator = 0
        # assume fixed leg is paid annually... just to keep things simple!
        for t in self.get_fixed_leg_payment_times():
            par_rate_denominator += np.exp(-zcb_curve.get_zero_rate(t)*t)
        
        return par_rate_numerator / par_rate_denominator