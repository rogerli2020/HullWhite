from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree, Node
from src.Swaption import EuropeanSwaption, SwaptionType
from src.HullWhiteTreeUtil import HullWhiteTreeUtil
import numpy as np

class HullWhiteTreeEuropeanSwaptionPricer:
    """
    Prices European swaptions using a Hull-White trinomial tree.
    """

    @staticmethod
    def _verify_timesteps(tree: OneFactorHullWhiteTrinomialTree, swaption: EuropeanSwaption) -> bool:
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
    def price(swaption: EuropeanSwaption, tree: OneFactorHullWhiteTrinomialTree) -> float:
        """
        Price a European swaption using the Hull-White tree.
        """

        # verify tree compatibility
        if not HullWhiteTreeEuropeanSwaptionPricer._verify_timesteps(tree, swaption):
            raise Exception("Swap cashflows are misaligned with the tree time steps.")

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