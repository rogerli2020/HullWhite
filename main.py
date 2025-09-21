from HullWhite import OneFactorHullWhiteModel
from TrinomialTree import OneFactorHullWhiteTrinomialTree

payment_times = [0, 1.5, 1.6, 2]
model = OneFactorHullWhiteModel(1, 0.3)
tree = OneFactorHullWhiteTrinomialTree(model, payment_times)

tree.build_tree()
tree.visualize_tree()