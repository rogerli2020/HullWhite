from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree, Node, LayerAttributesStruct
import numpy as np

class HullWhiteTreeUtil:

    @staticmethod
    def get_nodes_at_time(tree: OneFactorHullWhiteTrinomialTree, t: float) -> list[Node]:
        if t not in tree.t_to_layer:
            raise Exception(f"No layer found at time {t}.")
        layer = tree.t_to_layer[t]
        return tree.get_nodes_at_layer(layer)

    @staticmethod
    def get_zcb_price_dict(tree: OneFactorHullWhiteTrinomialTree, 
                           t0: float, T: float, maturity_node_values: dict={}) -> dict:
        if t0 not in tree.t_to_layer or T not in tree.t_to_layer:
            raise Exception(f"Invalid t0 or T for the given tree.")
        if t0 > T:
            raise Exception(f"t0 cannot be greater than or equal to T.")
        
        # set up layer data at T
        zcb_dict = {}
        for node in tree.get_nodes_at_layer(tree.t_to_layer[T]):
            if (node.layer_attr.layer_id, node.j) in maturity_node_values:
                # use the given maturity value if given.
                zcb_dict[ (node.layer_attr.layer_id, node.j) ] = maturity_node_values[ 
                    zcb_dict[ (node.layer_attr.layer_id, node.j) ] ]
            else:
                zcb_dict[ (node.layer_attr.layer_id, node.j) ] = 1

        # edge case
        if t0 == T:
            return zcb_dict

        # backward induction
        cur_layer: LayerAttributesStruct = tree.t_to_layer[T].prev_layer_attr
        stop: bool = False
        while cur_layer is not None:
            if stop:
                break
            stop = cur_layer is tree.t_to_layer[t0]
            delta_t: float = cur_layer.child_delta_t
            for node in tree.get_nodes_at_layer(cur_layer):
                cur_short_rate = node.value
                expected_zcb_price = 0
                for i in range(0, 3):
                    child_node: Node = node.children[i]
                    child_prob = node.children_prob[i]
                    child_price = zcb_dict[ (child_node.layer_attr.layer_id, child_node.j) ]
                    expected_zcb_price += child_price * child_prob * np.exp(-delta_t*cur_short_rate)
                zcb_dict[ (cur_layer.layer_id, node.j) ] = expected_zcb_price
            cur_layer = cur_layer.prev_layer_attr
        
        return zcb_dict
    
    @staticmethod
    def P(tree: OneFactorHullWhiteTrinomialTree, t0: float, T: float) -> float:
        return HullWhiteTreeUtil.price_zcb(tree, t0, T)
    
    @staticmethod
    def price_zcb(tree: OneFactorHullWhiteTrinomialTree, t0: float, T: float) -> float:
        zcb_prices = HullWhiteTreeUtil.get_zcb_price_dict(tree, t0, T)

        price = 0.0
        valuation_layer: LayerAttributesStruct = tree.t_to_layer[t0]
        for node in tree.get_nodes_at_layer(valuation_layer):
            ind = (valuation_layer.layer_id, node.j)
            Q = tree.node_lookup(*ind).Q
            node_zcb = zcb_prices[ind]
            price += node_zcb * Q
        
        return price
    
    @staticmethod
    def discount_cf_given_nodes_and_zcb_price_dict():
        ...