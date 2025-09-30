from HullWhiteTrinomialTree import OneFactorHullWhiteTrinomialTree, Node, LayerAttributesStruct

class HullWhiteTreeUtil:
    @staticmethod
    def get_nodes_at_time(tree: OneFactorHullWhiteTrinomialTree, t: float) -> list[Node]:
        if t not in tree.t_to_layer:
            raise Exception(f"No layer found at time {t}.")
        layer = tree.t_to_layer[t]
        return tree.get_nodes_at_layer(layer)

    @staticmethod
    def get_zcb_prices_from_start_to_end(tree: OneFactorHullWhiteTrinomialTree, start_t: float, end_t: float) -> dict[(int, int), float]:
        if start_t not in tree.t_to_layer:
            raise Exception(f"No layer found at time {start_t}.")
        if end_t not in tree.t_to_layer:
            raise Exception(f"No layer found at time {end_t}.")
        start_layer = tree.t_to_layer[start_t]
        end_layer = tree.t_to_layer[end_t]
        if start_layer.layer_id >= end_layer.layer_id:
            raise Exception(f"start_t must be less than end_t.")
        
        zcb_prices: dict[(int, int), float] = {}
        