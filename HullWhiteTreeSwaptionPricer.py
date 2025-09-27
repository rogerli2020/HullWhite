from OneFactorHullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from Swaption import Swaption

class HullWhiteTreeSwaptionPricer:
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

    def _verify_timesteps(self, swaption: Swaption):
        ...

    def price(self, swaption: Swaption) -> float:
        self._verify_timesteps(swaption)
        ...