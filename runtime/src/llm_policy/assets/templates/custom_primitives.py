"""Candidate-local JAX helpers shared by the two roles.

This module is optional. Keep helpers mechanics- or strategy-focused, pure,
fixed-shape, and JAX traceable. It may import JAX and the public ``llm_policy``
API, but not the environment, evaluator, workspace rollout scripts, or I/O.
"""
