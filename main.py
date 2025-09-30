from Swaption import EuropeanSwaption, SwaptionType
from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from HullWhite import OneFactorHullWhiteModel
from HullWhiteTreeSwaptionPricer import HullWhiteTreeEuropeanSwaptionPricer
from ZeroRateCurve import StepwiseZeroRateCurve, LinearZeroRateCurve
from HullWhiteTreeUtil import HullWhiteTreeUtil

hw_model = OneFactorHullWhiteModel(a=0.003, sigma=0.01)
zcb_curve = StepwiseZeroRateCurve()
swaption = EuropeanSwaption(
    swaption_type=SwaptionType.PAYER,
    expiry=2,
    swap_start=2,
    swap_end=6,
    payment_frequency=0.5,
    notional=1,
    strike=0.04,
    fixed=0.04,
)
tree = OneFactorHullWhiteTrinomialTree(hw_model, swaption.get_valuation_times(), zcb_curve, 0.5)

tree.build_tree()

d = HullWhiteTreeUtil.get_zcb_price_dict(tree, t0=swaption.swap_start, T=swaption.swap_end)
print(d)

# pricer = HullWhiteTreeEuropeanSwaptionPricer(tree)
# price = pricer.price(swaption)
# print(price)

# 2.123 %