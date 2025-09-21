from HullWhite import OneFactorHullWhiteModel
from TrinomialTree import OneFactorHullWhiteTrinomialTree
import random


payment_times = [0, 1.5, 1.6, 2]
model = OneFactorHullWhiteModel(0.01, 0.025)
tree = OneFactorHullWhiteTrinomialTree(model, payment_times)

tree.build_tree()
tree.visualize_tree()