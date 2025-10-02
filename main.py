from Swaption import EuropeanSwaption, SwaptionType
from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from HullWhite import OneFactorHullWhiteModel
from HullWhiteTreeSwaptionPricer import HullWhiteTreeEuropeanSwaptionPricer
from ZeroRateCurve import StepwiseZeroRateCurve
from HullWhiteTreeUtil import HullWhiteTreeUtil

hw_model = OneFactorHullWhiteModel(a=0.003, sigma=0.01)
zcb_curve = StepwiseZeroRateCurve()
swaption = EuropeanSwaption(
    swaption_type=SwaptionType.PAYER,
    expiry=1,
    swap_start=1,
    swap_end=6,
    payment_frequency=0.5,
    notional=1,
    strike=0.00,
    fixed=0.00,
)

swaption.set_ATM_strike_fixed_rate_and_strike(zcb_curve)

tree = OneFactorHullWhiteTrinomialTree(hw_model, 
                                       swaption.get_valuation_times(), zcb_curve, swaption.payment_frequency)

tree.build_tree(verbose=True)

pricer = HullWhiteTreeEuropeanSwaptionPricer(tree)
print(pricer.price(swaption))
