from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree, Node, LayerAttributesStruct
import numpy as np
from functools import wraps
from src.VectorizedHullWhiteTrinomialTree import VectorizedHW1FTrinomialTree, LayerInfoCols
from numba import njit

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
        """
        Given initial time t0 and maturity time T, get the expected ZCB prices of all nodes at t=t0 implied by the tree.
        Returns the vector of expected ZCB prices, unmasked.
        """
        tree_args = [
            tree.layer_information,
            tree.short_rate_tree,
            tree.node_mask_tree,
            tree.row_index,
            tree.p_up_tree,
            tree.p_mid_tree,
            tree.p_down_tree,
            tree.mid_index_tree,
            tree.j0_index
        ]
        return HullWhiteTreeUtil._get_zcb_price_vector_helper(*tree_args, t0, T)
    
    @staticmethod    
    @njit    
    def _get_zcb_price_vector_helper(tree_layer_information,
                                tree_short_rate_tree,
                                tree_node_mask_tree,
                                tree_row_index,
                                tree_p_up_tree,
                                tree_p_mid_tree,
                                tree_p_down_tree,
                                tree_mid_index_tree,
                                tree_j0_index,
                             t0: float, T: float) -> np.ndarray:

        # round times
        t0 = round(t0, 4)
        T = round(T, 4)

        # find row indexes
        t0_row_index: int|None  = None
        T_row_index: int|None   = None
        for layer_index in range(0, len(tree_layer_information)):
            if tree_layer_information[layer_index, 0] == t0:    # LayerInfoCols.T = 0
                t0_row_index = layer_index
            elif tree_layer_information[layer_index, 0] == T:
                T_row_index = layer_index

        # validations
        if t0 >= T:
            raise Exception(f"t0 cannot be greater than or equal to T.")
        if t0_row_index is None or T_row_index is None:
            raise Exception(f"Invalid t0 or T for the given tree.")

        # terminal price
        cols            = tree_short_rate_tree.shape[1]

        # keep track of things
        child_price     = np.ones(cols, dtype=float)    # P(T, T) = 1
        current_price   = np.zeros(cols, dtype=float)

        # backward induction step
        parent_layer_index = T_row_index-1  # start at parent
        up_child_contribution   = np.zeros(cols, dtype=float)
        mid_child_contribution  = np.zeros(cols, dtype=float)
        down_child_contribution = np.zeros(cols, dtype=float)
        while parent_layer_index >= t0_row_index:
            
            # layer wise information
            delta_t             = tree_layer_information[parent_layer_index, 1]

            # discount weighted by child probability!
            # parent layer
            parent_layer_mask   = tree_node_mask_tree[parent_layer_index]
            parent_rates        = tree_short_rate_tree[parent_layer_index][parent_layer_mask]
            parent_layer_indexes= tree_row_index[parent_layer_mask]

            # transient probabilities
            p_up                = tree_p_up_tree[parent_layer_index][parent_layer_mask]
            p_mid               = tree_p_mid_tree[parent_layer_index][parent_layer_mask]
            p_down              = tree_p_down_tree[parent_layer_index][parent_layer_mask]

            # child indexes
            _temp = tree_mid_index_tree[parent_layer_index] + tree_j0_index
            _temp_masked = _temp[parent_layer_mask]
            mid_child_indexes = _temp_masked.astype(np.int64)
            up_child_indexes    = (mid_child_indexes + 1).astype(np.int64)
            down_child_indexes  = (mid_child_indexes - 1).astype(np.int64)

            # E[P] = SIGMA SUM OVER i OF [ P_ci * p_ci * exp( -delta_t * parent_rate ) ]
            parent_instantaneous_discount   = np.exp( -delta_t * parent_rates )

            # reset these buffers
            up_child_contribution.fill(0.0)
            mid_child_contribution.fill(0.0)
            down_child_contribution.fill(0.0)

            # np.add.at(up_child_contribution,   parent_layer_indexes,   
            #           child_price[up_child_indexes]   * p_up    * parent_instantaneous_discount)
            # np.add.at(mid_child_contribution,  parent_layer_indexes,  
            #           child_price[mid_child_indexes]  * p_mid   * parent_instantaneous_discount)
            # np.add.at(down_child_contribution, parent_layer_indexes, 
            #           child_price[down_child_indexes] *p_down   * parent_instantaneous_discount)

            n = parent_layer_indexes.size
            for i in range(n):
                idx = parent_layer_indexes[i]

                # up child
                up_idx = up_child_indexes[i]
                up_child_contribution[idx] += child_price[up_idx] * p_up[i] * parent_instantaneous_discount[i]

                # mid child
                mid_idx = mid_child_indexes[i]
                mid_child_contribution[idx] += child_price[mid_idx] * p_mid[i] * parent_instantaneous_discount[i]

                # down child
                down_idx = down_child_indexes[i]
                down_child_contribution[idx] += child_price[down_idx] * p_down[i] * parent_instantaneous_discount[i]

            # add up
            current_price = up_child_contribution + mid_child_contribution + down_child_contribution

            # prepare for next iteration
            child_price = current_price.copy()
            current_price.fill(0.0)
            parent_layer_index -= 1
        
        return child_price


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