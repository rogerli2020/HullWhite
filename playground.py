import pandas as pd
import re
import numpy as np
from ZeroRateCurve import ExampleNSSCurve
from HullWhite import OneFactorHullWhiteModel
from Swaption import EuropeanSwaption, SwaptionType
from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from HullWhiteTreeSwaptionPricer import HullWhiteTreeEuropeanSwaptionPricer
from scipy.optimize import least_squares
from concurrent.futures import ThreadPoolExecutor, as_completed

hw_model = OneFactorHullWhiteModel(0.0075)
hw_model.set_constant_sigma(0.02)
valuation_times = [0,1]
zcb_curve = ExampleNSSCurve()
tree = OneFactorHullWhiteTrinomialTree(hw_model, valuation_times, zcb_curve, 0.083333)
tree.build_tree(verbose=False)
tree.visualize_tree()