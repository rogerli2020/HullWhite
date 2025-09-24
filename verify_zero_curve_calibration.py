
import numpy as np
from HullWhite import OneFactorHullWhiteModel
from TrinomialTree import OneFactorHullWhiteTrinomialTree, Node
from ZeroRateCurve import LinearZeroRateCurve

payment_times = [0, 26]
model = OneFactorHullWhiteModel(0.03, 0.005)
zcb_curve = LinearZeroRateCurve(0.0, 0.05, 30)

# set up and build tree
tree = OneFactorHullWhiteTrinomialTree(model, payment_times, zcb_curve, 2)
tree.build_tree()

def price_zcb_by_tree(tree: OneFactorHullWhiteTrinomialTree, maturity_time: float) -> float:
    """
    Compute the price of a zero-coupon bond maturing at `maturity_time`
    using the state prices (node.Q) in the tree.
    """
    # find the maturity layer
    cur_layer = tree.root_node.layer_attr
    while cur_layer is not None and abs(cur_layer.t - maturity_time) > 1e-12:
        cur_layer = cur_layer.next_layer_attr

    if cur_layer is None:
        raise ValueError(f"Maturity {maturity_time} not found in tree.payment_times")

    # sum of state prices at that layer = bond price
    nodes_at_layer = [
        node for (layer_id, _), node in tree._node_lookup.items()
        if layer_id == cur_layer.layer_id
    ]
    return sum(node.Q for node in nodes_at_layer)




cur_layer = tree.root_node.layer_attr
actual_price = []
tree_price = []

while cur_layer:
    m = cur_layer.layer_id
    P = np.exp( -1 * cur_layer.t * zcb_curve.get_zero_rate(cur_layer.t) )
    P_tree = price_zcb_by_tree(tree, cur_layer.t)
    actual_price.append(P)
    tree_price.append(P_tree)

    cur_layer = cur_layer.next_layer_attr

print(actual_price)
print(tree_price)

print(np.array(tree_price) - np.array(actual_price))

# Calibrated!!!!