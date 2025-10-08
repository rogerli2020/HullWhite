from src.HullWhite import OneFactorHullWhiteModel
from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from src.ZeroRateCurve import ExampleNSSCurve
from src.Swaption import EuropeanSwaption, SwaptionType
from src.HullWhiteTreeSwaptionPricer import HullWhiteTreeEuropeanSwaptionPricer

pricer = HullWhiteTreeEuropeanSwaptionPricer

swaption = EuropeanSwaption(
    SwaptionType.PAYER,
    swap_start=2,
    swap_end=7,
)

model = OneFactorHullWhiteModel(0.003)
model.set_sigmas_from_vector( [0.02]*12 )
zcb_curve = ExampleNSSCurve()
tree = swaption.build_valuation_tree(zcb_curve, set_ATM_strike=True, model=model, timestep=(1/24), verbose=True)

price_bps = pricer.price_in_bps(swaption, tree)
print(price_bps)