from HullWhite import OneFactorHullWhiteModel
from TrinomialTree import OneFactorHullWhiteTrinomialTree
from ZeroRateCurve import LinearZeroRateCurve

payment_times = [0, 1, 2, 3, 4, 5, 6, 7, 8]
model = OneFactorHullWhiteModel(0.2, 0.1)
tree = OneFactorHullWhiteTrinomialTree(model, payment_times, LinearZeroRateCurve(), 1)

tree.build_tree()