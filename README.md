
# One Factor Hull White Model in Python

## Introduction
This is a Python implementation of the one factor Hull-White (HW1F) model and a replication of [The General Hull-White Model and Super Calibration](https://archive.nyu.edu/handle/2451/26690) by Hull and White (August 2000).

## Usage Example
### 1. Building a Tree:
```Python
from  src.HullWhite  import  OneFactorHullWhiteModel
from  src.VectorizedHullWhiteTrinomialTree  import  VectorizedHW1FTrinomialTree
from  src.ZeroRateCurve  import  ExampleNSSCurve

model  =  OneFactorHullWhiteModel(0.003)
model.set_constant_sigma(0.3)
timestep  = 1/2

payment_times = [0, 0.5, 1, 1.5]
tree  =  VectorizedHW1FTrinomialTree(model,
payment_times=payment_times, zcb_curve=ExampleNSSCurve(), timestep=timestep)

tree.build()
```
This snippet of code would create a tree for a certain instrument with semi-annual payments (where the first payment happens exactly 6mo from now) and maturiting within 1.5 years. The timestep of the tree is set to 6mo, and the ZCB curve is set to a ```ZeroRateCurve``` object which must have a member function ```zero_rate_curve.get_zero_rate(t: float) -> float```.

### 2. Using a Built Tree:
The short rates at each node, the transient probabilities, and children indexes will be stored in arrays as follows. To filter out paddings at each layer, use the mask stored in the array ```tree.node_mask_tree```. These arrays allow for easy traversal of the tree and vectorized calculations using the tree.
```Python
mid_index_tree  =  tree.mid_index_tree  +  tree.j0_index  # translates the mid_index_tree to absolute indices for easy indexing.

print("Short rates at each node:")
print(tree.short_rate_tree.transpose())

print("Up child index and transition probabilities at each node:")
print((mid_index_tree  +  1).transpose())
print(tree.p_up_tree.transpose())

print("Mid child index and transition probabilities at each node:")
print((mid_index_tree).transpose())
print(tree.p_mid_tree.transpose())

print("Down child index and transition probabilities at each node:")
print((mid_index_tree  -  1).transpose())
print(tree.p_down_tree.transpose())

print("State prices at each node:")
print(tree.Q_tree.transpose())

print("Mask to identify node locations:")
print(tree.node_mask_tree.transpose())
```

Output:
```
Short rates at each node:
[[ 0.          0.          0.          0.        ]
 [ 0.          0.          0.         -0.97999137]
 [ 0.          0.         -0.66965222 -0.6125679 ]
 [ 0.         -0.33619346 -0.30222875 -0.24514444]
 [ 0.02051109  0.03123     0.06519471  0.12227902]
 [ 0.          0.39865346  0.43261817  0.48970248]
 [ 0.          0.          0.80004163  0.85712594]
 [ 0.          0.          0.          1.2245494 ]
 [ 0.          0.          0.          0.        ]]
Up child index and transition probabilities at each node:
[[5. 5. 5. 5.]
 [5. 5. 5. 2.]
 [5. 5. 3. 3.]
 [5. 4. 4. 4.]
 [5. 5. 5. 5.]
 [5. 6. 6. 6.]
 [5. 5. 7. 7.]
 [5. 5. 5. 8.]
 [5. 5. 5. 5.]]
[[0.         0.         0.         0.        ]
 [0.         0.         0.         0.16442679]
 [0.         0.         0.16517117 0.16517117]
 [0.         0.16591779 0.16591779 0.16591779]
 [0.16666667 0.16666667 0.16666667 0.16666667]
 [0.         0.16741779 0.16741779 0.16741779]
 [0.         0.         0.16817117 0.16817117]
 [0.         0.         0.         0.16892679]
 [0.         0.         0.         0.        ]]
Mid child index and transition probabilities at each node:
[[4. 4. 4. 4.]
 [4. 4. 4. 1.]
 [4. 4. 2. 2.]
 [4. 3. 3. 3.]
 [4. 4. 4. 4.]
 [4. 5. 5. 5.]
 [4. 4. 6. 6.]
 [4. 4. 4. 7.]
 [4. 4. 4. 4.]]
[[0.         0.         0.         0.        ]
 [0.         0.         0.         0.66664642]
 [0.         0.         0.66665767 0.66665767]
 [0.         0.66666442 0.66666442 0.66666442]
 [0.66666667 0.66666667 0.66666667 0.66666667]
 [0.         0.66666442 0.66666442 0.66666442]
 [0.         0.         0.66665767 0.66665767]
 [0.         0.         0.         0.66664642]
 [0.         0.         0.         0.        ]]
Down child index and transition probabilities at each node:
[[3. 3. 3. 3.]
 [3. 3. 3. 0.]
 [3. 3. 1. 1.]
 [3. 2. 2. 2.]
 [3. 3. 3. 3.]
 [3. 4. 4. 4.]
 [3. 3. 5. 5.]
 [3. 3. 3. 6.]
 [3. 3. 3. 3.]]
[[0.         0.         0.         0.        ]
 [0.         0.         0.         0.16892679]
 [0.         0.         0.16817117 0.16817117]
 [0.         0.16741779 0.16741779 0.16741779]
 [0.16666667 0.16666667 0.16666667 0.16666667]
 [0.         0.16591779 0.16591779 0.16591779]
 [0.         0.         0.16517117 0.16517117]
 [0.         0.         0.         0.16442679]
 [0.         0.         0.         0.        ]]
State prices at each node:
[[0.         0.         0.         0.        ]
 [0.         0.         0.         0.00768006]
 [0.         0.         0.03267382 0.07686478]
 [0.         0.16496614 0.23838194 0.2710971 ]
 [1.         0.65986457 0.48789938 0.38735009]
 [0.         0.16496614 0.19837571 0.18773935]
 [0.         0.         0.02262715 0.03686278]
 [0.         0.         0.         0.00255067]
 [0.         0.         0.         0.        ]]
Mask to identify node locations:
[[False False False False]
 [False False False  True]
 [False False  True  True]
 [False  True  True  True]
 [ True  True  True  True]
 [False  True  True  True]
 [False False  True  True]
 [False False False  True]
 [False False False False]]
```
### 3. Discounting using the Tree
Use the function ```HullWhiteTreeUtil.get_zcb_price_vector``` to get the expected ZCB price between ```t0``` and ```T``` for all nodes in layer where ```t = t0```.
```python
from  src.HullWhiteTreeUtil  import  HullWhiteTreeUtil
import  numpy  as  np

# get one year ZCB price at all nodes 6 months from now
root_node_layer_index, six_mo_layer_index  =  0, 1
one_yr_zcb_price_at_nodes_6mo_from_now  =  HullWhiteTreeUtil.get_zcb_price_vector(tree, t0=0.5, T=1.5)
print(one_yr_zcb_price_at_nodes_6mo_from_now[tree.node_mask_tree[six_mo_layer_index]])

# discount it back to today by taking the dot product with state prices
print(np.dot(one_yr_zcb_price_at_nodes_6mo_from_now, tree.Q_tree[six_mo_layer_index]))

# should be the same as the direct ZCB price from today to 1.5 years, or the ZCB price implied by the given zero rate curve.
print(HullWhiteTreeUtil.get_zcb_price_vector(tree, t0=0, T=1.5)[tree.node_mask_tree[root_node_layer_index]])
print(np.exp(-1.5  *  ExampleNSSCurve().get_zero_rate(1.5)))
```
Output:
```
[1.3841852  0.95830674 0.66346021]
0.9701448358716545
[0.97014484]
0.9701448358716543
```

### 4. Pricing and Calibration

Refer to ```LM_calibration.py``` and ```./src/HullWhiteTreeSwaptionPricer.py``` for examples of pricing simple European swaptions and calibrating the implied volatility structure to market observables.