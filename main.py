from Swaption import EuropeanSwaption, SwaptionType
from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from HullWhite import OneFactorHullWhiteModel
from HullWhiteTreeSwaptionPricer import HullWhiteTreeEuropeanSwaptionPricer
from ZeroRateCurve import StepwiseZeroRateCurve

hw_model = OneFactorHullWhiteModel(a=0.01, sigma=0.001)
zcb_curve = StepwiseZeroRateCurve()
swaption = EuropeanSwaption(
    swaption_type=SwaptionType.PAYER,
    expiry=1,
    swap_start=1,
    swap_end=6,
    payment_frequency=0.5,
    notional=1,
    strike=0.02123,
    fixed=0.02123,
    spread=0.00,
)
tree = OneFactorHullWhiteTrinomialTree(hw_model, swaption.get_valuation_times(), zcb_curve, 0.5)

tree.build_tree()
tree.visualize_tree()

pricer = HullWhiteTreeEuropeanSwaptionPricer(tree)
price = pricer.price(swaption)
print(price)

# 2.123 %