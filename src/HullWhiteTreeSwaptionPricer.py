from src.HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree, Node, LayerAttributesStruct
from src.Swaption import EuropeanSwaption, SwaptionType
from src.HullWhiteTreeUtil import HullWhiteTreeUtil

class HullWhiteTreeEuropeanSwaptionPricer:
    """
    Prices European swaptions using a Hull-White trinomial tree.

    Parameters
    ----------
    tree : OneFactorHullWhiteTrinomialTree
        A pre-built Hull-White trinomial tree.
    """

    def __init__(self, tree: OneFactorHullWhiteTrinomialTree) -> None:
        if not tree.tree_is_built():
            raise Exception("Tree must be built before pricing swaptions.")
        self.tree = tree

    def _verify_timesteps(self, swaption: EuropeanSwaption) -> bool:
        ts = swaption.get_valuation_times()
        tree_ts = set()
        cur_layer = self.tree.root_node.layer_attr
        while cur_layer is not None:
            tree_ts.add(cur_layer.t)
            cur_layer = cur_layer.next_layer_attr
        for t in ts:
            if t not in tree_ts:
                return False
        return True
    
    def price_in_bps(self, swaption: EuropeanSwaption) -> float:
        """
        Price a European swaption using the Hull-White tree and return the premium in basis points.
        """
        return self.price(swaption) * 10000

    def price(self, swaption):
        """
        Price a European swaption using the Hull-White tree.
        """

        if not self._verify_timesteps(swaption):
            raise Exception("Swap cashflows are misaligned with the tree time steps.")

        # 2️⃣ Identify expiry layer
        expiry_layer = self.tree.t_to_layer[swaption.swap_start]
        delta_t = expiry_layer.child_delta_t

        # 3️⃣ Compute ZCB prices at relevant nodes (helper function assumed)
        zcb_prices = HullWhiteTreeUtil.get_zcb_price_dict(
            self.tree, swaption.swap_start, swaption.swap_end
        )

        # 4️⃣ Initialize swaption payoff at expiry nodes
        for node in self.tree.get_nodes_at_layer(expiry_layer):

            # ---- Floating leg (par approximation) ----
            floating_leg = 1.0 - HullWhiteTreeUtil.get_node_specific_zcb_price(
                self.tree, node, swaption.swap_end
            )

            # ---- Fixed leg ----
            fixed_leg = 0.0
            for T in swaption.get_fixed_leg_payment_times():
                P_T = HullWhiteTreeUtil.get_node_specific_zcb_price(self.tree, node, T)
                fixed_leg += P_T  # assuming tau = 1
            fixed_leg *= swaption.fixed

            # ---- Option payoff ----
            swap_pv = floating_leg - fixed_leg
            if swaption.swaption_type == "PAYER":
                node.swaption_value = max(swap_pv, 0.0)
            elif swaption.swaption_type == "RECEIVER":
                node.swaption_value = max(-swap_pv, 0.0)
            else:
                raise Exception("Unknown swaption type")

        # 5️⃣ Backward induction to root
        cur_layer = expiry_layer
        while cur_layer != self.tree.root_node.layer_attr:
            prev_layer = cur_layer.previous_layer_attr
            dt = prev_layer.child_delta_t  # time step to next layer

            for parent_node in self.tree.get_nodes_at_layer(prev_layer):
                # Compute expected discounted value from children
                child_values = np.array([child.swaption_value for child in parent_node.children])
                child_Qs = np.array([child.Q for child in parent_node.children])
                r_parent = parent_node.value

                # discounted expectation
                parent_node.swaption_value = np.sum(child_Qs * np.exp(-r_parent * dt) * child_values)

            cur_layer = prev_layer

        # 6️⃣ Return value at root
        return self.tree.root_node.swaption_value