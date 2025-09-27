from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree, Node
from Swaption import EuropeanSwaption

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
        if not self._verify_timesteps(swaption):
            raise Exception("Underlying swap is incompatible with the given tree due to misaligned time gaps.")
        valuation_lookup = {}

        # get all the nodes at option expiry first.
        expiry_nodes: list[Node] = []
        node: Node
        for node in self.tree._node_lookup:
            if node.layer_attr.t == swaption.swap_start:
                expiry_nodes.append(node)
        
        # ...
        payment_time_set: set = set(swaption.get_valuation_times())
        payment_time_set.remove(0.0)
        payment_time_set.remove(swaption.swap_start)
        expiry_node_option_payoffs: list[tuple[Node, float]] = []
        for node in expiry_nodes:
            expiry_Q = node.Q
            cashflow_sum = 0.0
            current_node = node
            children = [child for child in node.children]
            while children:
                ...
            
            # only discount to parent... and assume parent is root node?
            cashflow_sum /= expiry_Q