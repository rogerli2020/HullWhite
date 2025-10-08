import numpy as np
from HullWhite import OneFactorHullWhiteModel
from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from ZeroRateCurve import ExampleNSSCurve
from HullWhiteTreeUtil import HullWhiteTreeUtil

payment_times = [0, 2]
model = OneFactorHullWhiteModel(0.03)
model.set_constant_sigma(0.02)
zcb_curve = ExampleNSSCurve()

# set up and build tree
tree = OneFactorHullWhiteTrinomialTree(model, payment_times, zcb_curve, 0.01)
tree.build_tree(verbose=True)

# loop
cur_layer = tree.root_node.layer_attr
closed_form_price = []
tree_price = []
while cur_layer:
    m = cur_layer.layer_id
    P = np.exp( -1 * cur_layer.t * zcb_curve.get_zero_rate(cur_layer.t) )
    P_tree = HullWhiteTreeUtil.P(tree, 0, cur_layer.t)
    closed_form_price.append(P)
    tree_price.append(P_tree)

    cur_layer = cur_layer.next_layer_attr

diff = np.array(tree_price) - np.array(closed_form_price)
diff = diff ** 2
diff = list([round(float(se), 8) for se in diff])
diff

print(diff)