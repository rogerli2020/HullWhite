from HullWhite import OneFactorHullWhiteModel
from OneFactorHullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from ZeroRateCurve import LinearZeroRateCurve

payment_times = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
model = OneFactorHullWhiteModel(0.002, 0.01)
tree = OneFactorHullWhiteTrinomialTree(model, payment_times, LinearZeroRateCurve(), .5)

tree.build_tree(verbose=False)
tree.visualize_tree()