"""SPARE: Sparsification Preserving Assertions, Reasoning and Evidence.

Budgeted justification-preserving sparsification (BJPS) of OWL knowledge graphs.
"""
from .kb import TBox, Reasoner, build_instance, closure_of, justifications, TYPE, THING
from .algorithms import (coverage, robust_coverage, total_weight, triple_greedy,
                         justification_greedy, robust_greedy, ilp, spare_hybrid)
from . import baselines, generators

__version__ = "0.1.0"
