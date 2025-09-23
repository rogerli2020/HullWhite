from HullWhite import OneFactorHullWhiteModel
from TrinomialTree import OneFactorHullWhiteTrinomialTree
from ZeroRateCurve import LinearZeroRateCurve

payment_times = [0, 1.5, 1.6, 2, 10]
model = OneFactorHullWhiteModel(1, 0.3)
tree = OneFactorHullWhiteTrinomialTree(model, payment_times, LinearZeroRateCurve(), 2)

tree.build_tree()
tree.visualize_tree()