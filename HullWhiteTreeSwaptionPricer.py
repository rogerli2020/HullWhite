from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree, Node, LayerAttributesStruct
from Swaption import EuropeanSwaption, SwaptionType

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

    def price_swap(self, swaption: EuropeanSwaption, valuation_lookup: dict={}) -> float:
        """
        Returns the PV of the underlying swap at its initiation.
        """
        if not self._verify_timesteps(swaption):
            raise Exception("Underlying swap is incompatible with the given tree due to misaligned time gaps.")
        raise NotImplementedError

    def price(self, swaption: EuropeanSwaption) -> float:

        # def discount_to_swap_start(value: float, current_node: Node, swap_start_layer: LayerAttributesStruct):
        #     current_parent = current_node.parent
        #     while current_parent.layer_attr != swap_start_layer:
        #         value /= current_parent.Q
        #         current_parent = current_parent.parent
        #     # discount one last time at swap start layer
        #     value /= current_parent.Q
        #     return value

        if not self._verify_timesteps(swaption):
            raise Exception("Underlying swap is incompatible with the given tree due to misaligned time gaps.")

        # get all the nodes at swap start first.
        swap_start_nodes: list[Node] = []
        node: Node
        for node in self.tree._node_lookup.values():
            if node.layer_attr.t == swaption.swap_start:
                swap_start_nodes.append(node)
        
        # price underlying swaps implied at each option expiry node
        payment_time_set: set = set(swaption.get_valuation_times())
        payment_time_set.remove(0.0)
        payment_time_set.remove(swaption.swap_start)
        # expiration_layer: LayerAttributesStruct = swap_start_nodes[0].layer_attr
        expiry_node_swap_price: list[tuple[Node, float]] = []
        price = 0
        for node in swap_start_nodes:
            swap_start_q = node.Q
            pv_swap = 0
            current_layer = set([node])
            next_layer = set()
            while len(current_layer) > 0:
                current_layer = list(current_layer)
                for cur_node in current_layer:
                    if current_layer[0].layer_attr.t in payment_time_set:
                        # there would be payments here, compute them!
                        cf_fixed = swaption.fixed * swaption.notional * swaption.payment_frequency  # fixed cf
                        r_float = node.value + swaption.spread                                      # floating rate
                        cf_float = r_float * swaption.notional * swaption.payment_frequency         # floating cf
                        cf_net = cf_float - cf_fixed                                                # net

                        # scale net payment by Q
                        node_contribution = cf_net * node.Q

                        # discount back
                        # node_contribution = discount_to_swap_start(node_contribution, cur_node, )

                        # add to total
                        pv_swap += node_contribution

                    # add children to the next layer regardless if current node has payments
                    for child in cur_node.children:
                        next_layer.add(child)
                
                # prepare for next iteration...
                current_layer = next_layer
                next_layer = set()

            # treat expiry node as the root node for swap prices
            pv_swap /= swap_start_q
            expiry_node_swap_price.append( (node, pv_swap) )

            # for each swap calculation result, calculate swaption payoff
            if swaption.swaption_type == SwaptionType.PAYER:
                payoff = max(pv_swap, 0.0)
            elif swaption.swaption_type == SwaptionType.RECEIVER:
                payoff = max(-pv_swap, 0.0)

            price += payoff * swap_start_q

        price_in_bps = price / swaption.notional * 100        
        return price_in_bps