from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree, Node
from src.VectorizedHullWhiteTrinomialTree import VectorizedHW1FTrinomialTree
from src.Swaption import EuropeanSwaption, SwaptionType
from src.HullWhiteTreeUtil import HullWhiteTreeUtil
import numpy as np
from numba import njit

class HullWhiteTreeEuropeanSwaptionPricer:
    """
    Prices European swaptions using a Hull-White trinomial tree.
    """

    @staticmethod
    def _verify_timesteps(tree: OneFactorHullWhiteTrinomialTree, swaption: EuropeanSwaption) -> bool:
        if isinstance(tree, VectorizedHW1FTrinomialTree):
            # keep things simple for now!
            return True
        else:
            ts = swaption.get_valuation_times()
            tree_ts = set()
            cur_layer = tree.root_node.layer_attr
            while cur_layer is not None:
                tree_ts.add(cur_layer.t)
                cur_layer = cur_layer.next_layer_attr
            for t in ts:
                if t not in tree_ts:
                    return False
            return True
    
    @staticmethod
    def price_in_bps(swaption: EuropeanSwaption, tree: OneFactorHullWhiteTrinomialTree) -> float:
        """
        Price a European swaption using the Hull-White tree and return the premium in basis points.
        """
        return HullWhiteTreeEuropeanSwaptionPricer.price(swaption, tree) * 1e4
    
    @staticmethod
    # @njit
    def _price_vectorized(
                tree_short_rate_tree,
                tree_layer_information,
                tree_node_mask_tree,
                tree_row_index,
                tree_p_up_tree,
                tree_p_mid_tree,
                tree_p_down_tree,
                tree_mid_index_tree,
                tree_j0_index,
                tree_Q_tree,
                swaption_fixed_leg_payment_times,
                swaption_swap_start,
                swaption_fixed,
                is_payer,
               ) -> float:

        # declare query wise information
        option_expiry_layer_row_index: int      = -1
        option_expiry_layer_mask: np.ndarray
        t0: float   # option expiry time
        fixed_leg_payment_times: list[float]
        tree_dimension = tree_short_rate_tree.shape
        
        # get query wise information
        t0 = swaption_swap_start
        for row in range(tree_dimension[0]):
            if tree_layer_information[row, 0] == t0:     # LayerInfoCols.T == 0
                option_expiry_layer_row_index = row
        if option_expiry_layer_row_index == -1:
            raise Exception("Incompatible tree.")
        option_expiry_layer_mask = tree_node_mask_tree[option_expiry_layer_row_index]
        fixed_leg_payment_times = swaption_fixed_leg_payment_times

        # main loop
        fixed_leg_values_sum        = np.zeros(tree_dimension[1])
        swap_end_zcb_values         = np.zeros(tree_dimension[1])
        swap_end_zcb_values_checker = False
        for T in fixed_leg_payment_times:
            zcb_price_vector = HullWhiteTreeUtil._get_zcb_price_vector_helper(
                                    tree_layer_information,
                                    tree_short_rate_tree,
                                    tree_node_mask_tree,
                                    tree_row_index,
                                    tree_p_up_tree,
                                    tree_p_mid_tree,
                                    tree_p_down_tree,
                                    tree_mid_index_tree,
                                    tree_j0_index,
                                    t0,
                                    T
                                )
            zcb_prices = zcb_price_vector
            fixed_leg_values_sum += zcb_prices

            # for floating leg value estimation
            if T == swaption_swap_start:
                swap_end_zcb_values = zcb_prices.copy()
                swap_end_zcb_values_checker = True

        # if somehow didn't reach T=swap_end, do it now... (is this check redundant?)
        if not swap_end_zcb_values_checker:
            zcb_price_vector = HullWhiteTreeUtil._get_zcb_price_vector_helper(
                                    tree_layer_information,
                                    tree_short_rate_tree,
                                    tree_node_mask_tree,
                                    tree_row_index,
                                    tree_p_up_tree,
                                    tree_p_mid_tree,
                                    tree_p_down_tree,
                                    tree_mid_index_tree,
                                    tree_j0_index,
                                    swaption_swap_start,
                                    T
                                )            
            zcb_prices = zcb_price_vector
            swap_end_zcb_values = zcb_prices.copy()
            swap_end_zcb_values_checker = True

        # calculations...
        fixed_leg_values    = (fixed_leg_values_sum * swaption_fixed)[option_expiry_layer_mask]
        floating_leg_values = 1.0 - swap_end_zcb_values[option_expiry_layer_mask]
        payer_swap_values   = fixed_leg_values - floating_leg_values
        payer_swap_values   = payer_swap_values
        
        # calculate option payoffs
        expiry_node_option_values: np.array
        if is_payer:
            expiry_node_option_values = np.maximum(payer_swap_values, 0.0)
        else:   # assume is receiver
            expiry_node_option_values = np.maximum(-payer_swap_values, 0.0)
        
        # Q is respective to valuation time (current time)
        expiry_Qs = tree_Q_tree[option_expiry_layer_row_index][option_expiry_layer_mask]

        # return dot product
        return np.dot(expiry_Qs, expiry_node_option_values)
              
        
    @staticmethod
    def price(swaption: EuropeanSwaption, tree: OneFactorHullWhiteTrinomialTree) -> float:
        """
        Price a European swaption using the Hull-White tree.
        """

        # verify tree compatibility
        if not HullWhiteTreeEuropeanSwaptionPricer._verify_timesteps(tree, swaption):
            raise Exception("Swap cashflows are misaligned with the tree time steps.")

        if isinstance(tree, VectorizedHW1FTrinomialTree):
            tree_args = [
                tree.short_rate_tree,
                tree.layer_information,
                tree.node_mask_tree,
                tree.row_index,
                tree.p_up_tree,
                tree.p_mid_tree,
                tree.p_down_tree,
                tree.mid_index_tree,
                tree.j0_index,
                tree.Q_tree,
            ]
            swaption_args = [
                swaption.get_fixed_leg_payment_times(),
                swaption.swap_start,
                swaption.fixed,
                swaption.swaption_type == SwaptionType.PAYER
            ]

            return HullWhiteTreeEuropeanSwaptionPricer._price_vectorized(
                *tree_args, *swaption_args
            )
        
        else:
            # get necessary data
            expiry_layer = tree.t_to_layer[swaption.swap_start]
            expiry_nodes: list[Node] = tree.get_nodes_at_layer(expiry_layer)
            fixed_leg_payment_times: list[float] = swaption.get_fixed_leg_payment_times()
            t0: float = expiry_layer.t # option expiry time
            n_nodes: int = len(expiry_nodes)

            fixed_leg_values_sum        = np.zeros(n_nodes)
            swap_end_zcb_values         = np.zeros(n_nodes)
            swap_end_zcb_values_checker = False
            for T in fixed_leg_payment_times:
                # first, get the ZCB prices of future payment
                # get_zcb_price_dict() call is expensive!!!
                zcb_price_dict = HullWhiteTreeUtil.get_zcb_price_dict(tree, t0, T)
                zcb_prices = np.array([zcb_price_dict[node] for node in expiry_nodes])

                # fixed leg values
                fixed_leg_values_sum += zcb_prices
                
                # if at swap end time, start calculating floating leg value (approximation)
                if T == swaption.swap_end:
                    swap_end_zcb_values = zcb_prices.copy()
                    swap_end_zcb_values_checker = True
            
            # if somehow didn't reach T=swap_end, do it now
            if not swap_end_zcb_values_checker:
                zcb_price_dict = HullWhiteTreeUtil.get_zcb_price_dict(tree, t0, swaption.swap_end)
                zcb_prices = np.array([zcb_price_dict[node] for node in expiry_nodes])
                swap_end_zcb_values = zcb_prices.copy()
                swap_end_zcb_values_checker = True
            
            # calculations...
            fixed_leg_values    = fixed_leg_values_sum * swaption.fixed
            floating_leg_values = 1.0 - swap_end_zcb_values
            payer_swap_values   = fixed_leg_values - floating_leg_values

            # calculate option payoffs
            expiry_node_option_values: np.array
            if swaption.swaption_type == SwaptionType.PAYER:
                expiry_node_option_values = np.maximum(payer_swap_values, 0.0)
            elif swaption.swaption_type == SwaptionType.RECEIVER:
                expiry_node_option_values = np.maximum(-payer_swap_values, 0.0)
            else:
                raise Exception("Unknown swaption type")
            
            # Q is respective to valuation time (current time)
            expiry_Qs = np.array([node.Q for node in expiry_nodes])

            # return dot product
            return np.dot(expiry_Qs, expiry_node_option_values)