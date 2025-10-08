from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree, Node, LayerAttributesStruct
import numpy as np
from functools import wraps

# decorator for rounding list of floats to 4 digits...
def round_list_floats(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, (list, tuple)):
            rounded = [round(x, 4) if isinstance(x, float) else x for x in result]
            return type(result)(rounded)
        return result
    return wrapper

class HullWhiteTreeUtil:

    @staticmethod
    def get_zcb_price_dict(tree: OneFactorHullWhiteTrinomialTree, 
                           t0: float, T: float, maturity_node_values: dict={}) -> dict[Node, float]:
        t0 = round(t0, 4)
        T = round(T, 4)
        if t0 not in tree.t_to_layer or T not in tree.t_to_layer:
            print(t0, T, tree.t_to_layer.keys())
            raise Exception(f"Invalid t0 or T for the given tree.")
        if t0 > T:
            raise Exception(f"t0 cannot be greater than or equal to T.")
        
        # set up layer data at T
        zcb_dict = {}
        for node in tree.get_nodes_at_layer(tree.t_to_layer[T]):
            if (node.layer_attr.layer_id, node.j) in maturity_node_values:
                # use the given maturity value if given.
                zcb_dict[node] = maturity_node_values[zcb_dict[node]]
            else:
                zcb_dict[node] = 1.0

        # edge case
        if t0 == T:
            return zcb_dict

        # backward induction
        cur_layer: LayerAttributesStruct    = tree.t_to_layer[T].prev_layer_attr
        last_layer: LayerAttributesStruct   = tree.t_to_layer[t0]
        stop: bool = False
        while cur_layer is not None:
            if stop:
                break
            stop = last_layer
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

    # @staticmethod
    # def get_nodes_at_time(tree: OneFactorHullWhiteTrinomialTree, t: float) -> list[Node]:
    #     if t not in tree.t_to_layer:
    #         raise Exception(f"No layer found at time {t}.")
    #     layer = tree.t_to_layer[t]
    #     return tree.get_nodes_at_layer(layer)

    # @staticmethod
    # def P(tree: OneFactorHullWhiteTrinomialTree, t0: float, T: float) -> float:
    #     t0 = round(t0, 4)
    #     T = round(T, 4)
    #     return HullWhiteTreeUtil.price_zcb(tree, t0, T)
    
    # @staticmethod
    # def price_zcb(tree: OneFactorHullWhiteTrinomialTree, t0: float, T: float) -> float:
    #     t0 = round(t0, 4)
    #     T = round(T, 4)
    #     zcb_prices = HullWhiteTreeUtil.get_zcb_price_dict(tree, t0, T)

    #     price = 0.0
    #     valuation_layer: LayerAttributesStruct = tree.t_to_layer[t0]
    #     for node in tree.get_nodes_at_layer(valuation_layer):
    #         ind = (valuation_layer.layer_id, node.j)
    #         Q = tree.node_lookup(*ind).Q
    #         node_zcb = zcb_prices[ind]
    #         price += node_zcb * Q
        
    #     return price
    
    # @staticmethod
    # def get_sum_of_discounted_cf_given_nodes(nodes, zcb_price_dict, notional):
    #     total = 0.0
    #     for node in nodes:
    #         m, j = node.layer_attr.layer_id, node.j
    #         zcb_price = zcb_price_dict[(m, j)]
    #         total += zcb_price * notional
    #     return total
    
    # @staticmethod
    # def get_node_specific_zcb_price(tree: OneFactorHullWhiteTrinomialTree, node: Node, T):
    #     T = round(T, 4)
    #     t0: float = node.layer_attr.t
    #     zcb_dict = HullWhiteTreeUtil.get_zcb_price_dict(tree, t0, T)
    #     return zcb_dict[ (node.layer_attr.layer_id, node.j) ]
    
    # @staticmethod
    # def calculate_state_prices(tree: OneFactorHullWhiteTrinomialTree, root_node: Node, 
    #                            terminal_layer: LayerAttributesStruct=None, inplace: bool=False):

    #     # Equation (5): Qij = SUM over k of p(i,j|i-1,k) * exp(-ri-1k * (ti - ti-1)) * Qi-1k
    #     Q_dict = {}
    #     cur_layer = root_node.layer_attr.next_layer_attr
    #     cur_delta_t = root_node.layer_attr.child_delta_t
    #     while cur_layer:
    #         for j in cur_layer.js:
    #             node_ij = tree.node_lookup(cur_layer.layer_id, j)
    #             Q_ij = 0.0
    #             for parent_node, cond_prob in node_ij.parents_to_conditional_prob.items():
    #                 r_parent = parent_node.value
    #                 Q_ij += cond_prob * np.exp(-r_parent * cur_delta_t) * parent_node.Q
    #             if inplace:
    #                 node_ij.Q = Q_ij
    #             else:
    #                 Q_dict[(cur_layer.layer_id, j)] = Q_ij
    #         if cur_layer is terminal_layer:
    #             break
    #         cur_delta_t = cur_layer.child_delta_t
    #         cur_layer = cur_layer.next_layer_attr
        
    #     return Q_dict