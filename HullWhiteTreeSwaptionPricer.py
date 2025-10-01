from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree, Node, LayerAttributesStruct
from Swaption import EuropeanSwaption, SwaptionType
from HullWhiteTreeUtil import HullWhiteTreeUtil

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

    def price(self, swaption: EuropeanSwaption) -> float:
        if not self._verify_timesteps(swaption):
            raise Exception("Underlying swap is incompatible with the given tree due to misaligned time gaps.")
        zcb_prices = HullWhiteTreeUtil.get_zcb_price_dict(self.tree, 
                                                          swaption.swap_start, swaption.swap_end)
