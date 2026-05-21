"""LEARN — the Hebbian update layer.

Hebbian co-activation on atoms, value updates on recalled atoms, steering-
vector nudges, counterfactual atom minting, distilled-apprentice training
tuples, taste signals — all flow into the lattice store from here. Wraps
the existing memories `learning.py` rules and extends them to all node
kinds.
"""
