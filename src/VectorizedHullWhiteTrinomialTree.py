from src.HullWhite import OneFactorHullWhiteModel
from src.ZeroRateCurve import ZeroRateCurve
import numpy as np
import pandas as pd

class VectorizedHullWhiteTrinomialTree:
    def __init__(self, model: OneFactorHullWhiteModel, payment_times: list[float], 
                 zcb_curve: ZeroRateCurve, timestep: float, desc: str="unnamed") -> None:
        self.model : OneFactorHullWhiteModel = model
        self.payment_times : list[float] = payment_times
        self.zcb_curve : ZeroRateCurve = zcb_curve
        self.timestep : float = timestep
        self.desc : str = desc

        self._build_timesteps()

        # tree specific
        self.tree: pd.DataFrame|None = None

        # validations
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

    def is_built(self) -> bool:
        return self.tree is not None
    
    def build(self) -> None:

        # layer wise information
        ts = np.array(self.payment_times)
        delta_ts = np.diff(self.payment_times + [self.payment_times[-1] + self.timestep])
        sigmas = np.array([self.model.sigma(t) for t in self.payment_times])
        delta_xs = sigmas * np.sqrt(3.0 * delta_ts)

        # merge them into a DataFrame
        layer_information = pd.DataFrame({
            'time': ts,
            'delta_t': np.append(delta_ts),
            'sigma': np.append(sigmas),
            'delta_x': np.append(delta_xs)
        })

        # build tree layer by layer
        short_rates_by_layer = [ np.array([0.0]) ]
        
        # initialize a data structure that links parent to its three children js layer by layer, to be filled up later.
        parent_to_children_js: list[list[tuple[int, int, int]]] = [ [ [1, 0, -1] ] ]  # first layer has only one node with j=0

        # traverse through layer_information
        max_j = 0
        for idx, row in layer_information.iterrows():
            t = row['time']
            delta_t = row['delta_t']
            sigma = row['sigma']
            delta_x = row['delta_x']

            # skip first layer
            if t == 0.0: continue

            # get highest j index for this layer
            j_max_this_layer: int = 0
            for r_parent in short_rates_by_layer[-1]:
                E_x = r_parent -r_parent * -self.model.a * delta_t
                m_i = round(E_x / delta_x)
                absmax: int = abs(max([m_i -1, m_i, m_i +1]))
                if absmax > j_max_this_layer:
                    j_max_this_layer = absmax

            # update max_j
            max_j = max(max_j, j_max_this_layer)

            # generate short rates for this layer from j_max to -j_max
            short_rates_this_layer = np.array([j * delta_x for j in range(-j_max_this_layer, j_max_this_layer + 1)])
            short_rates_by_layer.append(short_rates_this_layer)
        
        # short rates by layer is ready, create a 2d array with NaNs
        num_layers = len(short_rates_by_layer)
        tree_array = np.full((num_layers, 2 * max_j + 1), np.nan)
        j_offset = max_j  # to convert j index to array index
        for i, rates in enumerate(short_rates_by_layer):
            j_start = j_offset - (len(rates) // 2)
            tree_array[i, j_start:j_start + len(rates)] = rates
