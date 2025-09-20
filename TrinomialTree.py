from HullWhite import OneFactorHullWhiteModel
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

class Cashflow:
    def __init__(self):
        pass

    def get_valuation_date():
        pass

    def get_payment_dates():
        pass

    def get_valuation_and_payment_dates():
        pass


class BranchingProbabilities:
    def __init__(self) -> None:
        self._up : float = None
        self._mid : float = None
        self._down : float = None
        self._is_set : bool = False
    
    def set(self, up : float, mid : float, down : float) -> None:
        if abs((up + mid + down) - 1) > 0.0001:
            raise Exception("Sum of probabilities must be 1.")
        self._up = up
        self._mid = mid
        self._down = down
        self._is_set = True
    
    def get(self) -> tuple[float]:
        if not self._is_set:
            raise Exception("Branching probabilities not set.")
        return (self._up, self._mid, self._down)


class Node:
    def __init__(self, value : float, prob : float, j : int) -> None:
        self.value : float = value
        self.prob : float = prob
        self.j : int = j
        self.children : list = []
        self.children_prob : list[float] = []


class OneFactorHullWhiteTrinomialTree:
    def __init__(self, model : OneFactorHullWhiteModel, cashflow : Cashflow=None, k : int = 3) -> None:
        self.model : OneFactorHullWhiteModel = model
        self.cashflow : Cashflow = cashflow
        self.k : int = k
        self.root_node : Node = None

    def calculate_delta_x_annual(self) -> None:
        self._delta_x = self.model.sigma * np.sqrt(3)   # Equation (2)
    
    def build_tree(self):
        # relevant_dates : list[float] = self.cashflow.get_valuation_and_payment_dates()
        relevant_dates = [0, 1.5, 1.6, 2]
        if len(relevant_dates) <= 1: # relevant_dates at least includes the valuation date and maturity date.
            raise Exception("Too few dates.")
        
        # start by initiating the first node on the valuation date.
        parent_layer_date = relevant_dates[0]
        root_node = Node(0.025, 1, 0)
        current_parent_layer : list[Node] = [root_node]
        current_child_layer : list[Node] = []

        # iterate through all t
        for child_layer_date in relevant_dates[1:]:
            
            # compute delta t for the child layer
            delta_t : float = (child_layer_date - parent_layer_date)
            assert delta_t > 0

            # compute delta x for the child layer
            delta_x : float = self.model.sigma * np.sqrt(3 * delta_t)       # Eqn (2)

            # compute variance for the child layer
            V : float = self.model.sigma**2 * delta_t       # Footnote (3)
            assert V >= 0

            # iterate through parent layer
            for parent in current_parent_layer:
                # deterministic increment
                M = - parent.j * delta_x * self.model.a * delta_t

                # expected child value
                x_expected = parent.value + M

                # middle child index
                m_i = round(x_expected / delta_x)

                # potential child nodes and values
                children_j = [m_i - 1, m_i, m_i + 1]
                children_values = [j_c * delta_x for j_c in children_j]

                # transition probabilities
                p_up   = 0.5 * ((V + M**2) / (delta_x**2) + M / delta_x)
                p_mid  = 1 - (V + M**2) / (delta_x**2)
                p_down = 0.5 * ((V + M**2) / (delta_x**2) - M / delta_x)
                probs = [p_down, p_mid, p_up]

                # recombine
                for val_c, j_c, prob_c in zip(children_values, children_j, probs):
                    existing_node = None
                    for n in current_child_layer:
                        if n.j == j_c:
                            existing_node = n
                            break
                    if existing_node:
                        parent.children.append(existing_node)
                        parent.children_prob.append(prob_c)
                    else:
                        node = Node(value=val_c, j=j_c, prob=parent.prob * prob_c)
                        current_child_layer.append(node)
                        parent.children.append(node)
                        parent.children_prob.append(prob_c)
                
            # prepare for new iteration
            parent_layer_date = child_layer_date
            current_parent_layer = current_child_layer
            current_child_layer = []

        self.root_node = root_node

    def visualize_tree(self):
        """
        Visualize the Hull-White trinomial tree using matplotlib
        """
        if self.root_node is None:
            print("Tree is empty. Build the tree first.")
            return

        # Collect layers: list of lists of nodes
        layers = []
        current_layer = [self.root_node]
        while current_layer:
            layers.append(current_layer)
            next_layer = []
            for node in current_layer:
                next_layer.extend(node.children)
            current_layer = next_layer if next_layer else None

        # Plotting
        plt.figure(figsize=(12, 6))
        for i, layer in enumerate(layers):
            for node in layer:
                x = i  # layer index
                y = node.value
                plt.scatter(x, y, s=200, c='skyblue', zorder=3)
                plt.text(x, y, f"({node.j},{node.value:.2f})", ha='center', va='bottom', fontsize=8)

                # Draw edges to children
                for child in node.children:
                    plt.plot([x, i+1], [y, child.value], 'k-', alpha=0.4, zorder=1)

        plt.xlabel("Time Step / Layer")
        plt.ylabel("x value")
        plt.title("Hull-White Trinomial Tree Visualization")
        plt.grid(True, alpha=0.3)
        plt.show()