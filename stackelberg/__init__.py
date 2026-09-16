"""Shared implementation package for the Stackelberg game solvers."""

from .randomness import DEFAULT_SEED, get_global_seed, make_rng, set_global_seed

__all__ = ["DEFAULT_SEED", "get_global_seed", "make_rng", "set_global_seed"]

