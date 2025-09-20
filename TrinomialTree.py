from HullWhite import OneFactorHullWhiteModel
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class Node:
    def __init__(self, value : float, prob : float, j : int) -> None:
        self.value : float = value
        self.prob : float = prob
        self.j : int = j
        self.children : list = []
        self.children_prob : list[float] = []


class OneFactorHullWhiteTrinomialTree:
    def __init__(self, model : OneFactorHullWhiteModel, payment_times : list[float], k : int = 3) -> None:
        self.model : OneFactorHullWhiteModel = model
        self.payment_times : list[float] = payment_times
        self.k : int = k
        self.root_node : Node = None

        # basic validations
        assert len(self.payment_times) >= 2, "At least two payment times are required."
        assert all(t2 > t1 for t1, t2 in zip(self.payment_times, self.payment_times[1:])), "Payment times must be in ascending order."

    def build_tree(self):
        # start by initiating the first node on the valuation date.
        parent_layer_date = self.payment_times[0]
        root_node = Node(0.025, 1, 0) # TODO: hardcoded initial value
        self.root_node = root_node
        current_parent_layer : list[Node] = [root_node]
        current_child_layer : list[Node] = []

        # iterate through all t
        for child_layer_date in self.payment_times[1:]:
            
            # compute delta t for the child layer
            delta_t : float = (child_layer_date - parent_layer_date)
            assert delta_t >= 0, "Delta t must be non-negative."

            # compute delta x for the child layer
            delta_x : float = self.model.sigma * np.sqrt(3 * delta_t)       # Eqn (2)

            # compute variance for the child layer
            V : float = self.model.sigma**2 * delta_t                       # Footnote (3)
            assert V >= 0, "Variance must be non-negative."

            # iterate through parent layer
            for parent in current_parent_layer:
                # deterministic increment
                M = - parent.j * delta_x * self.model.a * delta_t           # Footnote (3)

                # expected child value
                x_expected = parent.value + M                               # Current node value + deterministic increment

                # middle child index
                m_i = int(x_expected // delta_x) 

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

        print("Tree built successfully.")

    def visualize_tree(self):
        """
        Cheap visualization of the Hull-White trinomial tree (no labels, batched drawing).
        """
        if self.root_node is None:
            print("Tree is empty. Build the tree first.")
            return

        # Collect layers
        layers = []
        current_layer = [self.root_node]
        while current_layer:
            layers.append(current_layer)
            next_layer = []
            for node in current_layer:
                next_layer.extend(node.children)
            current_layer = next_layer if next_layer else None

        # Prepare data
        xs, ys, lines = [], [], []
        for i, layer in enumerate(layers):
            for node in layer:
                x, y = i, node.value
                xs.append(x)
                ys.append(y)
                for child in node.children:
                    lines.append([(x, y), (i+1, child.value)])

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))

        # Draw all edges in one call
        if lines:
            lc = LineCollection(lines, colors="k", alpha=0.3, linewidths=0.5)
            ax.add_collection(lc)

        # Draw all nodes in one call
        ax.scatter(xs, ys, s=10, c="skyblue", alpha=0.8)

        ax.set_xlabel("Time Step / Layer")
        ax.set_ylabel("x value")
        ax.set_title("Hull-White Trinomial Tree (cheap visualization)")
        ax.grid(True, alpha=0.2)
        plt.show()