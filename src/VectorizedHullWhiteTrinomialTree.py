from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from enum import IntEnum


class LayerInfoCols(IntEnum):
    T                   = 0
    DELTA_T             = 1
    SIGMA               = 2
    LN_ACTUAL_ZCB_PRICE = 3
    DELTA_X             = 4
    V                   = 5
    COMPONENT_1         = 6


class VectorizedHW1FTrinomialTree(OneFactorHullWhiteTrinomialTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # vecotorized version specific
        self.tree_built: bool   = False
        self.j0_index: int|None = None
        self.layer_information: pd.DataFrame
        self.short_rate_tree: np.ndarray
        self.node_mask_tree: np.ndarray
        self.mid_j_tree: np.ndarray
        self.p_up_tree: np.ndarray
        self.p_mid_tree: np.ndarray
        self.p_down_tree: np.ndarray
        self.Q_tree: np.ndarray
    
    def tree_is_built(self):
        return self.tree_built
    
    def build_tree(self) -> None:

        def transform_lists_of_array_to_single_array(l: list[np.ndarray], 
                                                     tree: np.ndarray):
            rows, cols = tree.shape
            for row in range(rows):
                arr = l[row]
                arr_len = len(arr)
                padding_len = (cols - arr_len) // 2

                for j in range(padding_len):
                    tree[row, j] = 0.0
                for j in range(arr_len):
                    tree[row, padding_len + j] = arr[j]
                for j in range(padding_len + arr_len, cols):
                    tree[row, j] = 0.0

            return tree
    
        def reposition_and_sum(original_values, new_index, cols):
            result = np.zeros(cols, dtype=float)
            np.add.at(result, new_index.astype(int), original_values)
            return result
        
        def create_reachable_mask_tree(m_is_by_layer, reachable_mask_tree):
            for row, m_is in enumerate(m_is_by_layer):
                m_is += self.j0_index
                if row == 0:
                    reachable_mask_tree[0][self.j0_index] = 1
                if not row+1 > reachable_mask_tree.shape[0]-1:
                    reachable_mask_tree[row+1][m_is+1] = 1
                    reachable_mask_tree[row+1][m_is] = 1
                    reachable_mask_tree[row+1][m_is-1] = 1
                m_is -= self.j0_index
            return reachable_mask_tree

        print(f"Building tree {self.desc}...")

        # just to make things handy
        a: float = self.model.a
        last_t: float = self.payment_times[-1]

        # layer wise data. Each layer represents a timestep
        ts = np.array(self.payment_times)
        delta_ts = np.diff(self.payment_times + [self.payment_times[-1] + self.timestep])
        next_ts = ts + delta_ts
        sigmas = np.array([self.model.sigma(t) for t in ts])
        next_rs = np.array([self.zcb_curve.get_zero_rate(t) for t in next_ts])
        ln_actual_zcb_prices = -next_rs*next_ts
        delta_xs = sigmas * np.sqrt(3.0 * delta_ts)
        Vs = sigmas * sigmas * delta_ts
        component_1s = Vs / delta_xs / delta_xs  # to avoid redundant calculation

        # merge layer information into a dataframe just to make things clearer
        # layer_information = pd.DataFrame({
        #     't': ts,
        #     'delta_t': delta_ts,
        #     'sigma': sigmas,
        #     'ln_actual_zcb_price': ln_actual_zcb_prices,
        #     'delta_x': delta_xs,
        #     'V': Vs,
        #     'component_1': component_1s
        # })
        layer_information = np.array(
            [
                ts,
                delta_ts,
                sigmas,
                ln_actual_zcb_prices,
                delta_xs,
                Vs,
                component_1s
            ]
        )
        layer_information = layer_information.transpose()

        # the goal is to translate the original OOP tree implementation into numpy vectors and arrays.
        # the short rate tree is to be represented by a 2D array with the following dimensions:
            # because numpy is row-major, and most operations happens on a per layer 
            # basis, rows represent timesteps.
            # and now, the 2d array can only be as wide as the maximum j across all layers.
            # and it is to be determined in the following steps.
        short_rate_tree_array_rows: int = len(self.payment_times)
        short_rate_tree_array_cols: int

        # essentially what connects a parent node to the child node is m_i. If we know m_i
        # we know all 3 children, as it will always be m_i+1, m_i, and m_i-1, thus knowing m_i is enough
        # to link all nodes together.

        # to determine m_i for each node, we still need to have all the short rates...
        # if we know the parent short rates, we know what js the child layer has. and the
        # child layer's short rates can then be calculated on a per layer, rather than per node basis.

        # so for now, compute and store short rates layer by layer, use the following data strcture
        # to hold the short rates and m_i for all nodes, layer by layer.
        # furthermore, we need a data structure to hold transient probabilities
        short_rates_by_layer: list[np.array] = [ np.array([0.0]) ]
        m_is_by_layer: list[np.array] = []
        m_is_by_layer: list[np.array] = []
        p_up_by_layer: list[np.array] = []
        p_mid_by_layer: list[np.array] = []
        p_down_by_layer: list[np.array] = []
        mask_by_layer: list[np.array] = []

        # to know the short rates for the next layer, we need to know the range of js for the layer.
        # finally, we also need to keep track of the maximum j across all layers so we know the shape
        # of the 2d array representing the entire tree.

        # ==============================================================
        # STEP 1: Build preliminary tree structure
        # ==============================================================

        # traverse layer by layer...
        # TODO: maybe this could be done faster by precomputing the number of cols?
        maximum_j_across_all_layers: int = 0
        for layer_index in range(0, len(layer_information)):
            delta_t     = layer_information[layer_index, LayerInfoCols.DELTA_T]
            delta_x     = layer_information[layer_index, LayerInfoCols.DELTA_X]
            component_1 = layer_information[layer_index, LayerInfoCols.COMPONENT_1]

            # get m_i of each node in this layer, and also get the maximum_j_this_layer
            parent_layer_short_rates: np.array  = short_rates_by_layer[-1]
            E_x: np.array                       = (parent_layer_short_rates - 
                                                   self.model.a * parent_layer_short_rates * delta_t)
            this_layer_m_is: np.array           = np.round(E_x / delta_x).astype(int)
            maximum_j_this_layer: int           = np.max(np.abs(this_layer_m_is)) + 1

            # now you have the parent layer's m_is, append it
            m_is_by_layer.append(this_layer_m_is)
            mask_by_layer.append(np.ones(len(this_layer_m_is)))

            # and calculate transient probabilities here...
            x_expected: np.array    = parent_layer_short_rates - parent_layer_short_rates * a * delta_t
            alpha_down = (x_expected - (this_layer_m_is - 1) * delta_x) / delta_x
            alpha_mid  = (x_expected - this_layer_m_is * delta_x) / delta_x
            alpha_up   = (x_expected - (this_layer_m_is + 1) * delta_x) / delta_x
            p_up   = 0.5 * (component_1 + alpha_up**2 + alpha_up)
            p_down = 0.5 * (component_1 + alpha_down**2 - alpha_down)
            p_mid  = 1 - component_1 - alpha_mid**2
            p_up_by_layer.append(p_up)
            p_mid_by_layer.append(p_mid)
            p_down_by_layer.append(p_down)

            # update maximum_j across all layers:
            maximum_j_across_all_layers = max(maximum_j_across_all_layers, maximum_j_this_layer)

            # now compute short rates for the children, and append to the list
            short_rates_by_layer.append(delta_x* np.arange(-maximum_j_this_layer, 
                                                           maximum_j_this_layer + 1, dtype=int))
            
        # now we're ready to create the 2d array of short rate tree!
        short_rate_tree_array_cols  = maximum_j_across_all_layers * 2 + 1
        self.j0_index               = maximum_j_across_all_layers

        # declare the trees' shape in memory
        short_rate_tree         = np.zeros((short_rate_tree_array_rows, short_rate_tree_array_cols))
        short_rate_tree_m_i     = np.zeros((short_rate_tree_array_rows, short_rate_tree_array_cols))
        mask_tree               = np.zeros((short_rate_tree_array_rows, short_rate_tree_array_cols))
        p_up_tree               = np.zeros((short_rate_tree_array_rows, short_rate_tree_array_cols))
        p_mid_tree              = np.zeros((short_rate_tree_array_rows, short_rate_tree_array_cols))
        p_down_tree             = np.zeros((short_rate_tree_array_rows, short_rate_tree_array_cols))
        reachable_mask_tree     = np.zeros((short_rate_tree_array_rows, short_rate_tree_array_cols))
        Q_tree                  = np.zeros((short_rate_tree_array_rows, short_rate_tree_array_cols))

        # get mask trees
        mask_tree           = transform_lists_of_array_to_single_array(mask_by_layer, mask_tree).astype(bool)
        reachable_mask_tree = create_reachable_mask_tree(m_is_by_layer, reachable_mask_tree).astype(bool)

        # fill the valid positions directly
        short_rate_tree[mask_tree]      = np.concatenate(short_rates_by_layer[:-1])
        short_rate_tree_m_i[mask_tree]  = np.concatenate(m_is_by_layer)
        p_up_tree[mask_tree]            = np.concatenate(p_up_by_layer)
        p_mid_tree[mask_tree]           = np.concatenate(p_mid_by_layer)
        p_down_tree[mask_tree]          = np.concatenate(p_down_by_layer)

        print(f"Calibrating tree {self.desc} to ZCB curve... Array shape: {short_rate_tree.shape}")

        #region Calibrate to ZCB
        # ==============================================================
        # STEP 2: Calibrate to ZCB yields
        # formulas: https://www.math.hkust.edu.hk/~maykwok/courses/MAFS525/Topic4_4.pdf
        # ==============================================================

        Q_tree[0, self.j0_index] = 1.0
        for layer_index in range(0, len(layer_information)):
            ln_actual_zcb_price = layer_information[layer_index, LayerInfoCols.LN_ACTUAL_ZCB_PRICE]
            delta_t             = layer_information[layer_index, LayerInfoCols.DELTA_T]
            t                   = layer_information[layer_index, LayerInfoCols.T]

            # calculate adjustment
            layer_mask  = reachable_mask_tree[layer_index]
            Q           = Q_tree[layer_index][layer_mask]
            short_rates = short_rate_tree[layer_index]
            a           = np.log(Q) - short_rates[layer_mask] * delta_t
            log_sum     = logsumexp(a)
            adjustment  = (log_sum - ln_actual_zcb_price) / delta_t

            # apply adjustment
            short_rate_tree[layer_index][layer_mask] += adjustment
            parent_short_rates = short_rate_tree[layer_index][layer_mask]

            # prepare for next iteration by calculating the Qs for the children
            if t < last_t:

                # parent
                Q_parent            = Q

                # child js
                mid_child_j         = short_rate_tree_m_i[layer_index][layer_mask]

                # child indexes
                mid_child_index     = mid_child_j + self.j0_index
                up_child_index      = mid_child_index + 1
                down_child_index    = mid_child_index - 1

                # child transient probabilities
                p_up                = p_up_tree[layer_index][layer_mask]
                p_mid               = p_mid_tree[layer_index][layer_mask]
                p_down              = p_down_tree[layer_index][layer_mask]

                # stepwise discount factor
                stepwise_df         = np.exp( -parent_short_rates*delta_t )

                # calculate Qs by k
                Q_up    = (Q_parent * p_up) * stepwise_df
                Q_mid   = (Q_parent * p_mid) * stepwise_df
                Q_down  = (Q_parent * p_down) * stepwise_df

                # resize arrays
                Q_up    = reposition_and_sum(Q_up,  up_child_index, 
                                             short_rate_tree_array_cols)
                Q_mid   = reposition_and_sum(Q_mid, mid_child_index, 
                                             short_rate_tree_array_cols)
                Q_down  = reposition_and_sum(Q_down,down_child_index, 
                                             short_rate_tree_array_cols)

                # add them all up!
                Q_tree[layer_index+1] = Q_up + Q_mid + Q_down
                
        #endregion

        # assign values
        self.j0_index: int|None = None
        self.layer_information = layer_information
        self.short_rate_tree = short_rate_tree
        self.node_mask_tree = reachable_mask_tree
        self.mid_index_tree = short_rate_tree_m_i
        self.p_up_tree = p_up_tree
        self.p_mid_tree = p_mid_tree
        self.p_down_tree = p_down_tree
        self.Q_tree = Q_tree
        self.tree_built = True

        print(f"Tree {self.desc} built! Array shape: {short_rate_tree.shape}")