from enum import Enum
from src.ZeroRateCurve import ZeroRateCurve
from src.HullWhite import OneFactorHullWhiteModel
from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from src.VectorizedHullWhiteTrinomialTree import VectorizedHW1FTrinomialTree
from src.ZeroRateCurve import ZeroRateCurve
from src.HullWhiteTreeUtil import round_list_floats
import numpy as np

class SwaptionType(Enum):
    PAYER = "payer"
    RECEIVER = "receiver"

class EuropeanSwaption:
    def __init__(self, swaption_type: SwaptionType,
                 swap_start: float, swap_end: float, expiry: float=None,
                 payment_frequency: float=0.5, notional: float=1, strike: float=0.0, 
                 fixed: float=0.0) -> None:
        self.swaption_type = swaption_type
        self.swap_start = swap_start
        self.swap_end = swap_end
        self.payment_frequency = payment_frequency
        self.notional = notional
        self.strike = strike
        self.fixed = fixed
        if expiry is None:
            expiry = self.swap_start
    
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
        """
        Calculates the swap par rate using the par rate formula
        """
        par_rate_denominator: float
        par_rate_numerator: float

        par_rate_numerator = ( np.exp(-zcb_curve.get_zero_rate(t=self.swap_start)*self.swap_start)
                              - np.exp(-zcb_curve.get_zero_rate(t=self.swap_end)*self.swap_end))

        zcb_prices = np.array([-zcb_curve.get_zero_rate(t)*t for t in self.get_fixed_leg_payment_times()])
        zcb_prices = np.exp(zcb_prices)
        par_rate_denominator = np.sum(zcb_prices)
        
        return par_rate_numerator / par_rate_denominator
    
    def build_valuation_tree_OLD(self, zcb_curve: ZeroRateCurve, set_ATM_strike: bool, 
                              model: OneFactorHullWhiteModel, timestep: float=None, 
                              verbose: bool=False) -> OneFactorHullWhiteTrinomialTree:
        timestep = self.payment_frequency if timestep is None else timestep
        if set_ATM_strike:
            self.set_ATM_strike_fixed_rate_and_strike(zcb_curve)
        tree = OneFactorHullWhiteTrinomialTree(model, self.get_valuation_times(), 
                                               zcb_curve, timestep, desc=self.__repr__())
        tree.build_tree(verbose=verbose)
        return tree
    
    def build_valuation_tree(self, zcb_curve: ZeroRateCurve, set_ATM_strike: bool, 
                              model: OneFactorHullWhiteModel, timestep: float=None, 
                              verbose: bool=False) -> VectorizedHW1FTrinomialTree:
        timestep = self.payment_frequency if timestep is None else timestep
        if set_ATM_strike:
            self.set_ATM_strike_fixed_rate_and_strike(zcb_curve)
        tree = VectorizedHW1FTrinomialTree(model, self.get_valuation_times(), 
                                               zcb_curve, timestep, desc=self.__repr__())
        tree.build()
        return tree
    
    def __repr__(self):
        return f"European Swaption {self.swap_start}Y{self.swap_end}Y"