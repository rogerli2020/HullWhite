from HullWhite import OneFactorHullWhiteModel
from TrinomialTree import OneFactorHullWhiteTrinomialTree
from ZeroRateCurve import LinearZeroRateCurve

payment_times = [0, .25, 1.25, 2.25, 3.25]
model = OneFactorHullWhiteModel(0.2, 0.1)
tree = OneFactorHullWhiteTrinomialTree(model, payment_times, LinearZeroRateCurve(), 0.25)

tree.build_tree()