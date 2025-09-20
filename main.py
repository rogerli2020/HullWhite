from HullWhite import OneFactorHullWhiteModel
from TrinomialTree import OneFactorHullWhiteTrinomialTree

model = OneFactorHullWhiteModel(0.25, 0.50)
tree = OneFactorHullWhiteTrinomialTree(model, None, 3)

tree.build_tree()
tree.visualize_tree()