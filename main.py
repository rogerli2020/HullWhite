from src.HullWhite import OneFactorHullWhiteModel
from src.ZeroRateCurve import ExampleNSSCurve
from src.VectorizedHullWhiteTrinomialTree import VectorizedHW1FTrinomialTree
from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from src.HullWhiteTreeUtil import HullWhiteTreeUtil
import numpy as np

model = OneFactorHullWhiteModel(0.03)
model.set_constant_sigma(0.3)

for _ in range(100):
    payment_times, timestep = [0, 1.6, 1.7, 2], 2
    payment_times, timestep = [0, 30], 1/12
    tree = VectorizedHW1FTrinomialTree(model, 
                                            payment_times=payment_times, zcb_curve=ExampleNSSCurve(), timestep=timestep)
    tree.build()

    print(
        HullWhiteTreeUtil.get_zcb_price_vector(tree, 0, 30)[tree.j0_index]
    )


# # # Tree unnamed:   Constructed layer 1440  for t=30.0      Number of nodes: 1657
# tree = OneFactorHullWhiteTrinomialTree(model, 
#                                         payment_times=payment_times, zcb_curve=ExampleNSSCurve(), timestep=timestep)
# tree.build_tree(verbose=False)

# print(HullWhiteTreeUtil.get_zcb_price_dict(tree, 0, 1.7))

# for key, val in tree.t_to_layer.items():
#     nodes = tree.get_nodes_at_layer(val)
#     print(np.array([node.value for node in nodes]))

print("Done!")