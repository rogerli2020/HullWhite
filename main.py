from src.HullWhite import OneFactorHullWhiteModel
from src.ZeroRateCurve import ExampleNSSCurve
from src.VectorizedHullWhiteTrinomialTree import VectorizedHW1FTrinomialTree
from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from src.HullWhiteTreeUtil import HullWhiteTreeUtil
from src.HullWhiteTreeSwaptionPricer import HullWhiteTreeEuropeanSwaptionPricer
from src.Swaption import EuropeanSwaption, SwaptionType
import numpy as np

model = OneFactorHullWhiteModel(0.03)
model.set_constant_sigma(0.3)
timestep = 1/48

# payment_times, timestep = [0, 1.6, 1.7, 2], 2
# payment_times, timestep = [0, 30], 1/12
# tree = VectorizedHW1FTrinomialTree(model, 
#                                         payment_times=payment_times, zcb_curve=ExampleNSSCurve(), timestep=timestep)
# tree.build()


# tree2 = OneFactorHullWhiteTrinomialTree(model, 
#                                         payment_times=payment_times, zcb_curve=ExampleNSSCurve(), timestep=timestep)
# tree2.build_tree(verbose=False)


swaption = EuropeanSwaption(
    swaption_type=SwaptionType.PAYER,
    swap_start=0.5,
    swap_end=20.5,
    payment_frequency=0.5,
)


# tree_OOP            = swaption.build_valuation_tree(ExampleNSSCurve(), True, model, timestep, True)
# price_oop           = HullWhiteTreeEuropeanSwaptionPricer.price_in_bps(swaption, tree_OOP)
# print(f"OOP Price:\t{price_oop}")

tree_VECTORIZED     = swaption.build_valuation_tree_vectorized(ExampleNSSCurve(), True, model, timestep, False)
price_vectorized    = HullWhiteTreeEuropeanSwaptionPricer.price_in_bps(swaption, tree_VECTORIZED)
print(f"Vec Price:\t{price_vectorized}")


print("Done!")