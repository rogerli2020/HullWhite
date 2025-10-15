from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree, Node, LayerAttributesStruct
import numpy as np
from functools import wraps
from src.VectorizedHullWhiteTrinomialTree import VectorizedHW1FTrinomialTree

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
    def get_zcb_price_vector(tree: VectorizedHW1FTrinomialTree,
                             t0: float, T: float) -> np.ndarray:
        
        # round times
        t0 = round(t0, 4)
        T = round(T, 4)

        # find row indexes
        t0_row_index: int|None  = None
        T_row_index: int|None   = None
        for idx, layer_info in tree.layer_information.iterrows():
            if layer_info['t'] == t0:
                t0_row_index = idx
            if layer_info['T'] == T:
                T_row_index = idx

        # validations
        if t0 >= T:
            raise Exception(f"t0 cannot be greater than or equal to T.")
        if t0_row_index is None or T_row_index is None:
            raise Exception(f"Invalid t0 or T for the given tree.")

        # 
        # # backward induction
        # cur_layer: LayerAttributesStruct    = tree.t_to_layer[T].prev_layer_attr
        # last_layer: LayerAttributesStruct   = tree.t_to_layer[t0]
        # stop: bool = False
        # while cur_layer is not None:
        #     if stop:
        #         break
        #     stop = cur_layer == last_layer
        #     delta_t: float = cur_layer.child_delta_t
        #     for node in tree.get_nodes_at_layer(cur_layer):
        #         cur_short_rate = node.value
        #         expected_zcb_price = 0
        #         for i in range(0, 3):
        #             child_node: Node = node.children[i]
        #             child_prob = node.children_prob[i]
        #             child_price = zcb_dict[child_node]
        #             expected_zcb_price += child_price * child_prob * np.exp(-delta_t*cur_short_rate)
        #         zcb_dict[node] = expected_zcb_price
        #     cur_layer = cur_layer.prev_layer_attr


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
            stop = cur_layer == last_layer
            delta_t: float = cur_layer.child_delta_t
            for node in tree.get_nodes_at_layer(cur_layer):
                cur_short_rate = node.value
                expected_zcb_price = 0
                for i in range(0, 3):
                    child_node: Node = node.children[i]
                    child_prob = node.children_prob[i]
                    child_price = zcb_dict[child_node]
                    expected_zcb_price += child_price * child_prob * np.exp(-delta_t*cur_short_rate)
                zcb_dict[node] = expected_zcb_price
            cur_layer = cur_layer.prev_layer_attr
        
        return zcb_dict