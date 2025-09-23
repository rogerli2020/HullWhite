from HullWhite import OneFactorHullWhiteModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dataclasses import dataclass
from ZeroRateCurve import ZeroRateCurve
import heapq

EPSILON = 1e-5

@dataclass
class LayerAttributesStruct:
    layer_id : int = 0
    t : float = 0
    child_delta_t : float = 0
    num_nodes : int = 0
    delta_x : float = 0
    next_layer_attr = None


class Node:
    def __init__(self, value : float, prob : float, j : int, layer_attr : LayerAttributesStruct) -> None:
        self.value : float = value
        self.prob : float = prob
        self.j : int = j
        self.children : list = []
        self.children_prob : list[float] = []
        self.layer_attr = layer_attr

        layer_attr.num_nodes += 1


class OneFactorHullWhiteTrinomialTree:
    """
    Trinomial tree data structure for the one-factor Hull-White interest rate model.

    Parameters
    ----------
    model : OneFactorHullWhiteModel
        A Hull-White model instance.
    payment_times : list of float
        A strictly increasing list of times (in years or the chosen time unit), 
        starting with the valuation date (typically 0) followed by the future payment dates.
    zcb_curve : ZeroRateCurve
        Object of any class representing the ZCB curve with method get_zero_rate(t: float)

    Methods
    -------
    build_tree()
        Builds the trinomial tree based on the model parameters.
    visualize_tree()
        Visualizes the structure of the trinomial tree.
    tree_is_built()
        Returns True if tree is built, False otherwise.
    """
    def __init__(self, model : OneFactorHullWhiteModel, payment_times : list[float], zcb_curve : ZeroRateCurve, timestep : float) -> None:
        self.model : OneFactorHullWhiteModel = model
        self.payment_times : list[float] = payment_times
        self.zcb_curve : ZeroRateCurve = zcb_curve
        self.timestep : float = timestep
        self.root_node : Node = None
        
        self._node_lookup : dict[tuple, Node] = {}
        self._build_timesteps()

        # sanity checks
        assert len(self.payment_times) >= 2, "At least two payment times are required."
        assert all(t2 > t1 for t1, t2 in zip(self.payment_times, self.payment_times[1:])), "Payment times must be in ascending order."
    
    def _build_timesteps(self):
        ttm = self.payment_times[-1]
        min_num_timesteps = int(ttm // self.timestep) + 1
        timesteps = [i * self.timestep for i in range(min_num_timesteps)]
        self.payment_times = list(dict.fromkeys(heapq.merge(timesteps, self.payment_times)))

    def tree_is_built(self) -> bool:
        """
        Returns True if tree is built, False otherwise.
        """
        return self.root_node is not None

    def build_tree(self):
        """
        Builds the tree based on the given model and payment times.
        """

        def qkj(m : int, k : int, j : int) -> float:
            target_node = self._node_lookup[ (m+1), j ]
            for key, val in enumerate(self._node_lookup[ (m, k) ].children):
                if val == target_node:
                    return self._node_lookup[ (m, k) ].children_prob[key]
            return 0
    
        # start by initiating the first node on the valuation date. Initial variance is 0 since f(r)|t=0 is deterministic.
        parent_layer_date = self.payment_times[0]
        root_layer_attr = LayerAttributesStruct(0, 0, self.payment_times[1])
        root_node = Node(0, 1, 0, root_layer_attr)
        self._node_lookup[(0, 0)] = root_node
        self.root_node = root_node
        current_parent_layer : list[Node] = [root_node]
        current_child_layer : list[Node] = []

        # lambdas for Equation (4)
        alpha : float       = lambda k     : (x_expected - k * delta_x) / delta_x
        p_up : float        = lambda alpha : 0.5 * (component_1 + alpha * alpha + alpha)
        p_down : float      = lambda alpha : 0.5 * (component_1 + alpha * alpha - alpha)
        p_mid : float       = lambda alpha : 1 - component_1 - alpha * alpha
 
        # iterate through all t
        for child_layer_date in self.payment_times[1:]:
            
            # compute delta t for the child layer
            delta_t : float = (child_layer_date - parent_layer_date)
            assert delta_t >= 0, "Delta t must be positive."

            # this is purely for graphing
            child_layer_attr = LayerAttributesStruct(
                layer_id=(current_parent_layer[0].layer_attr.layer_id+1),
                t=child_layer_date,
                child_delta_t=self.timestep,
                num_nodes=0
            )
            current_parent_layer[0].layer_attr.next_layer_attr = child_layer_attr

            # compute delta x for the child layer
            delta_x : float = self.model.sigma * np.sqrt(3 * delta_t)       # Equation (2)
            current_parent_layer[0].layer_attr.delta_x = delta_x

            # compute variance for the child layer
            V : float = self.model.sigma**2 * delta_t                       # Footnote (3)
            assert V >= 0, "Variance must be positive."

            # prevent redundant calculations
            component_1 : float = V / delta_x / delta_x                     # for Equation (4)

            # iterate through parent layer
            for parent in current_parent_layer:
                # just for graphing/visualization
                parent.layer_attr.child_delta_t = delta_t

                # deterministic mean reverting drift
                M = - parent.value * self.model.a * delta_t                 # Footnote (3)

                # expected child value
                x_expected = parent.value + M

                # middle child index (from 3. Choosing the branching process)
                m_i = round(x_expected / delta_x)
        
                # potential child nodes and values
                children_j = [m_i - 1, m_i, m_i + 1]
                children_values = [j_c * delta_x for j_c in children_j]

                # transition probabilities
                probs = [
                    p_down(alpha(m_i-1)),
                    p_mid(alpha(m_i)),
                    p_up(alpha(m_i+1)),
                ]

                # sanity check!
                assert abs(sum(probs) - 1) < EPSILON, "Sum of transition probabilities is not 1."

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
                        node = Node(value=val_c, j=j_c, prob=parent.prob * prob_c, layer_attr=child_layer_attr)
                        self._node_lookup[ (child_layer_attr.layer_id, node.j) ] = node
                        current_child_layer.append(node)
                        parent.children.append(node)
                        parent.children_prob.append(prob_c)
                
            # prepare for new iteration
            parent_layer_date = child_layer_date
            current_parent_layer = current_child_layer
            current_child_layer = []

        # Step 4. Adjusting the Tree to ZCB yields
        Q_lookup, alpha_lookup = {(0,0):1}, {0:0}
        current_layer : LayerAttributesStruct = self.root_node.layer_attr
        while current_layer is not None:
            m : int = current_layer.layer_id
            delta_R = current_layer.delta_x

            # Step 1: Calculate alpha_m
            alpha_m = np.log(sum(
                [
                    Q_lookup[(m,j)]*np.exp(-j*delta_R*current_layer.child_delta_t) for j in range(
                        -current_layer.num_nodes//2+1, current_layer.num_nodes//2+1) 
                ]
            ))
            p_m1 = np.exp( -self.zcb_curve.get_zero_rate(current_layer.t) * current_layer.t)
            alpha_m -= np.log(p_m1)
            alpha_m /= current_layer.child_delta_t
            alpha_lookup[m] = alpha_m

            # Step 2: Calculate Q for next layer
            if current_layer.next_layer_attr is not None:
                for j in range(-current_layer.next_layer_attr.num_nodes//2+1,
                            current_layer.next_layer_attr.num_nodes//2+1):
                    current_sum = 0
                    for k in range(-current_layer.num_nodes//2+1, current_layer.num_nodes//2+1):
                        current_val = (Q_lookup[ (m, k) ] 
                                    * qkj(m, k, j) 
                                    * np.exp((-alpha_m+k*(delta_R))
                                                *current_layer.child_delta_t))
                        current_sum += current_val
                    Q_lookup[(m+1,j)] = current_sum

            current_layer = current_layer.next_layer_attr

        current_layer : LayerAttributesStruct = self.root_node.layer_attr
        while current_layer is not None:
            m = current_layer.layer_id
            for k in range(-current_layer.num_nodes//2+1, current_layer.num_nodes//2+1):
                self._node_lookup[(m, k)].value += alpha_lookup[m]
            current_layer = current_layer.next_layer_attr

        print("Tree built successfully.")


    def visualize_tree(self):
        """
        ChatGPT: visualization of the Hull-White trinomial tree (no labels, batched drawing).
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
                node : Node
                x, y = node.layer_attr.t, node.value
                xs.append(x)
                ys.append(y)
                for child in node.children:
                    lines.append(
                        [
                            (x, y), 
                            (node.layer_attr.t + node.layer_attr.child_delta_t, child.value)
                        ]
                    )

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
        ax.set_title("Hull-White Trinomial Tree")
        ax.grid(True, alpha=0.2)
        plt.show()