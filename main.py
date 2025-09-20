from HullWhite import OneFactorHullWhiteModel
from TrinomialTree import OneFactorHullWhiteTrinomialTree
import random


payment_times = [0, 1]
for _ in range(10):
    payment_times.append(payment_times[-1] + random.uniform(0, 2))
model = OneFactorHullWhiteModel(0.25, 58888)
tree = OneFactorHullWhiteTrinomialTree(model, payment_times, 3)

tree.build_tree()
tree.visualize_tree()