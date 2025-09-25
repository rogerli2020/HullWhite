from OneFactorHullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree
from Swaption import Swaption

class HullWhiteTreeSwaptionPricer:
    """
    Prices European swaptions using a built Hull-White trinomial tree.

    Parameters
    ----------
    tree : OneFactorHullWhiteTrinomialTree
        A pre-built Hull-White trinomial tree.
    """

    def __init__(self, tree: OneFactorHullWhiteTrinomialTree) -> None:
        if not tree.tree_is_built():
            raise Exception("Tree must be built before pricing swaptions.")
        self.tree = tree

    def price(self, swaption: Swaption) -> float:
        """
        Prices a swaption using backward induction on the trinomial tree.
        """
        raise NotImplementedError("Unknown settlement type.")