from src.HullWhite import OneFactorHullWhiteModel
from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from src.ZeroRateCurve import ExampleNSSCurve

payment_times = [0, 30]
model = OneFactorHullWhiteModel(0.003)
model.set_sigmas_from_vector([0.0020] * 3 + [0.0030] * 3 + [0.0040] * 3 + [0.0050] * 3)
zcb_curve = ExampleNSSCurve()

tree = OneFactorHullWhiteTrinomialTree(model, payment_times, zcb_curve, 0.25, desc="Example")
tree.build_tree(verbose=True)