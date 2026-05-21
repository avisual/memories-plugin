"""DISTILL — Organ 8.

A small policy net (single transformer block + action head, <100M
params) trained continuously from successful traces. Within weeks it
handles the easy 80% of decisions without invoking the LLM at all.
"""
