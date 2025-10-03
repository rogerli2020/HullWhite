from Swaption import EuropeanSwaption, SwaptionType
from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from HullWhite import OneFactorHullWhiteModel
from HullWhiteTreeSwaptionPricer import HullWhiteTreeEuropeanSwaptionPricer
from ZeroRateCurve import ExampleLinearlyInterpolatedZeroRateCurve
from HullWhiteTreeUtil import HullWhiteTreeUtil

hw_model = OneFactorHullWhiteModel(a=0.003)
zcb_curve = ExampleLinearlyInterpolatedZeroRateCurve()
swaption = EuropeanSwaption(
    swaption_type=SwaptionType.PAYER,
    expiry=2,
    swap_start=2,
    swap_end=6,
    payment_frequency=0.5,
    notional=1,
    strike=0.00,
    fixed=0.00,
)

swaption.set_ATM_strike_fixed_rate_and_strike(zcb_curve)

# tree = OneFactorHullWhiteTrinomialTree(hw_model, swaption.get_valuation_times(), zcb_curve, swaption.payment_frequency)
tree = OneFactorHullWhiteTrinomialTree(hw_model, [0, 1.5, 1.6, 2], zcb_curve, 2)

tree.build_tree(verbose=True)
print(tree.t_to_layer)
tree.visualize_tree()

# pricer = HullWhiteTreeEuropeanSwaptionPricer(tree)
# print(pricer.price(swaption) * 10000)
