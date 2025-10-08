from HullWhite import OneFactorHullWhiteModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dataclasses import dataclass
from ZeroRateCurve import ZeroRateCurve
from dataclasses import dataclass, field

EPSILON = 1e-5

@dataclass
class LayerAttributesStruct:
    layer_id : int = 0
    t : float = 0
    child_delta_t : float = 0
    num_nodes : int = 0
    delta_x : float = 0
    next_layer_attr : "LayerAttributesStruct" = None
    prev_layer_attr : "LayerAttributesStruct" = None
    js: list[int] = field(default_factory=list)


class Node:
    def __init__(self, value: float,
                 j: int, layer_attr: LayerAttributesStruct) -> None:
        self.value : float = value
        self.j : int = j
        self.children : list = []
        self.children_prob : list[float] = []
        self.layer_attr = layer_attr
        self.Q : float = -1

        layer_attr.num_nodes += 1
        layer_attr.js.append(self.j)

        # for calculation purposes
        self.parents_to_conditional_prob : dict[Node, float] = {}

        self.index: tuple[int, int] = (layer_attr.layer_id, j)
    
    # for easy lookup
    def __eq__(self, other):
        return isinstance(other, Node) and self.index == other.index
    def __hash__(self):
        return hash(self.index)
    def __repr__(self):
        return f"Node({self.index}, {self.value})"

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
    def __init__(self, model: OneFactorHullWhiteModel, payment_times: list[float], 
                 zcb_curve: ZeroRateCurve, timestep: float, desc: str="") -> None:
        self.model : OneFactorHullWhiteModel = model
        self.payment_times : list[float] = payment_times
        self.zcb_curve : ZeroRateCurve = zcb_curve
        self.timestep : float = timestep
        self.desc : str = desc
        self.root_node : Node = None
        
        self.t_to_layer : dict[float, LayerAttributesStruct] = {}
        self._node_lookup : dict[tuple, Node] = {}
        self._build_timesteps()

        if not len(self.payment_times) >= 2:
            raise Exception("At least two payment times are required.")
        if not all(t2 > t1 for t1, t2 in zip(self.payment_times, self.payment_times[1:])):
            raise Exception("Payment times must be in ascending order.")
    
    def _build_timesteps(self):
        if not self.payment_times:
            raise ValueError("self.payment_times is empty.")
        if self.payment_times[0] != 0.0:
            raise ValueError("self.payment_times must start at 0.")

        new_times = []
        last_time = 0.0

        for pt in self.payment_times:
            t = last_time
            while t + self.timestep < pt:
                t += self.timestep
                new_times.append(round(t, 4))
            new_times.append(pt)
            last_time = pt
        self.payment_times = sorted(set([0.0] + new_times))
    
    def get_total_num_nodes(self) -> int:
        count = 0
        cur_layer = self.root_node.layer_attr
        while cur_layer:
            count += cur_layer.num_nodes
            cur_layer = cur_layer.next_layer_attr
        return count

    def tree_is_built(self) -> bool:
        """
        Returns True if tree is built, False otherwise.
        """
        return self.root_node is not None

    def build_tree(self, verbose=False):
        """
        Builds the tree based on the given model and payment times.
        """
    
        if verbose:
            print("Building tree...")

        # start by initiating the first node on the valuation date. Root node variance is zero.
        parent_layer_date = self.payment_times[0]
        root_layer_attr = LayerAttributesStruct(0, 0, self.payment_times[1])
        self.t_to_layer[parent_layer_date] = root_layer_attr
        root_node = Node(0, 0, root_layer_attr)
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

            # set up layer data
            child_layer_attr = LayerAttributesStruct(
                layer_id=(current_parent_layer[0].layer_attr.layer_id+1),
                t=child_layer_date,
                child_delta_t=self.timestep,
                num_nodes=0,
                prev_layer_attr=current_parent_layer[0].layer_attr
            )
            current_parent_layer[0].layer_attr.next_layer_attr = child_layer_attr
            self.t_to_layer[child_layer_date] = child_layer_attr

            # compute delta x for the child layer
            delta_x : float = self.model.sigma(parent_layer_date) * np.sqrt(3 * delta_t)       # Equation (2)
            current_parent_layer[0].layer_attr.next_layer_attr.delta_x = delta_x

            # compute variance for the child layer
            V : float = self.model.sigma(parent_layer_date)**2 * delta_t                       # Footnote (3)
            assert V >= 0, "Variance must be positive."

            # prevent redundant calculations
            component_1 : float = V / delta_x / delta_x                     # for Equation (4)

            # iterate through parent layer
            for parent in current_parent_layer:
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
                assert abs(sum(probs) - 1) < EPSILON

                # recombine
                for val_c, j_c, prob_c in zip(children_values, children_j, probs):

                    # in this loop, get or create the child node
                    child_node: Node

                    # if child with j already exists...
                    node_exists, child_node = self.node_lookup_safe(child_layer_attr.layer_id, j_c)

                    # else create a new child node and add to lookup
                    if not node_exists:
                        child_node = Node(value=val_c, j=j_c, 
                                    layer_attr=child_layer_attr)
                        current_child_layer.append(child_node)
                        self._node_lookup[(child_layer_attr.layer_id, j_c)] = child_node

                    # link parent and child
                    parent.children.append(child_node)
                    parent.children_prob.append(prob_c)
                    child_node.parents_to_conditional_prob[parent] = prob_c
            
            # simple print statement
            if verbose:
                print(
                    f"Tree {self.desc}\t"
                    f"Constructed layer {child_layer_attr.layer_id}\t"
                    f"for t={child_layer_attr.t}.\t"
                    f"Number of nodes: {child_layer_attr.num_nodes}.\t"
                )

            # prepare for new iteration
            parent_layer_date = child_layer_date
            current_parent_layer = current_child_layer
            current_child_layer = []

        #region Adjusting the Tree to match ZCB yields

        # ============================
        # calculate state prices
        # ============================
        self.calculate_state_prices(self.root_node, terminal_layer=None, inplace=True)




        #endregion

        if verbose:
            print(f"Tree {self.desc} built successfully.")
    
    def node_lookup(self, m: int, j: int) -> Node:
        if (m, j) not in self._node_lookup:
            raise Exception(f"No node found at ({m}, {j}).")
        return self._node_lookup[(m, j)]
    
    def node_lookup_safe(self, m: int, j: int) -> tuple:
        if (m, j) not in self._node_lookup:
            return False, None
        return True, self._node_lookup[(m, j)]

    def get_nodes_at_layer(self, layer: LayerAttributesStruct) -> list[Node]:
        nodes = []
        for j in layer.js:
            nodes.append(self.node_lookup(layer.layer_id, j))
        return nodes

    def calculate_state_prices(self, root_node: Node, 
                               terminal_layer: LayerAttributesStruct=None, inplace: bool=False):
        # Equation (5): Qij = SUM over k of p(i,j|i-1,k) * exp(-ri-1k * (ti - ti-1)) * Qi-1k
        Q_dict = {}
        Q_dict[(root_node.layer_attr.layer_id, root_node.j)] = 1.0
        if inplace:
            root_node.Q = 1.0
        cur_layer = root_node.layer_attr.next_layer_attr
        cur_delta_t = root_node.layer_attr.child_delta_t
        while cur_layer:
            for j in cur_layer.js:
                node_ij = self.node_lookup(cur_layer.layer_id, j)
                Q_ij = 0.0
                for parent_node, cond_prob in node_ij.parents_to_conditional_prob.items():
                    r_parent = parent_node.value
                    Q_ij += cond_prob * np.exp(-r_parent * cur_delta_t) * parent_node.Q
                if inplace:
                    node_ij.Q = Q_ij
                else:
                    Q_dict[(cur_layer.layer_id, j)] = Q_ij
            if cur_layer is terminal_layer:
                break
            cur_delta_t = cur_layer.child_delta_t
            cur_layer = cur_layer.next_layer_attr
        
        return Q_dict
    
    #region Visualization
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
    #endregion