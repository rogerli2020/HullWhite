from src.HullWhite import OneFactorHullWhiteModel
from src.ZeroRateCurve import ExampleNSSCurve
from src.VectorizedHullWhiteTrinomialTree import VectorizedHW1FTrinomialTree
from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
import numpy as np

model = OneFactorHullWhiteModel(0.03)
model.set_constant_sigma(0.3)

payment_times, timestep = [0, 1.6, 1.7, 2], 2
# payment_times, timestep = [0, 35], 1/32
tree = VectorizedHW1FTrinomialTree(model, 
                                        payment_times=payment_times, zcb_curve=ExampleNSSCurve(), timestep=timestep)
tree.build_tree()

# # # Tree unnamed:   Constructed layer 1440  for t=30.0      Number of nodes: 1657
# tree = OneFactorHullWhiteTrinomialTree(model, 
#                                         payment_times=payment_times, zcb_curve=ExampleNSSCurve(), timestep=timestep)
# tree.build_tree(verbose=True)

# for key, val in tree.t_to_layer.items():
#     nodes = tree.get_nodes_at_layer(val)
#     print(np.array([node.value for node in nodes]))

print("Done!")