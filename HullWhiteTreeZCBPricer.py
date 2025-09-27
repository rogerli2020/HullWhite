from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
import numpy as np

class HullWhiteTreeZCBPricer:
    """
    Kinda useless since you already have the ZCB curve... This is mainly for validation.
    """
    def __init__(self, tree: OneFactorHullWhiteTrinomialTree):
        self.tree = tree
        if not self.tree.tree_is_built():
            raise Exception("tree must be built before pricing.")

    def price(self, time_to_maturity: float, notional: float) -> float:
        # find the layer_id for the given ttm
        cur_layer = self.tree.root_node.layer_attr
        while cur_layer is not None and abs(cur_layer.t - time_to_maturity) > 1e-12:
            cur_layer = cur_layer.next_layer_attr
        if cur_layer is None: raise Exception("time to maturity not found on tree.")
        maturity_layer = cur_layer

        # for each j at maturity layer, set the price to notional in P_next
        # fill out cur_nodes as parent nodes 
        P_next = {}
        # cur_nodes = set()
        # for j in range(-maturity_layer.num_nodes//2+1, maturity_layer.num_nodes//2+1):
        #     P_next[j] = notional
        #     cur_nodes.add(self.tree._node_lookup[(maturity_layer.layer_id, j)].parent)
        for (m, j), node in self.tree._node_lookup.items():
            if m != maturity_layer.layer_id or node is None:
                continue
            P_next[j] = notional
            # cur_nodes.add(node.parent)

        # backward induction
        for m in range(maturity_layer.layer_id-1, -1, -1):
            P_cur = {}
            cur_nodes = []
            for (_m, j), node in self.tree._node_lookup.items():
                if _m == m:
                    cur_nodes.append(node)
            # next_cur_nodes = set()
            for node in cur_nodes:
                # for each node, get expected value if continued from current node, 
                # then, discount by the short rate represented by the node.
                continuation_value = sum(p * P_next[child.j] 
                                         for child, p in zip(node.children, node.children_prob))
                P_cur[node.j] = continuation_value*np.exp(-node.value*node.layer_attr.child_delta_t)
                # next_cur_nodes.add(node.parent)
            P_next = P_cur
            # cur_nodes = next_cur_nodes
        
        # root node's j is 0
        return P_next[0]