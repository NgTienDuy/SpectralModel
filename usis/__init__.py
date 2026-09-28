"""USIS - Universal Spectral Intelligence System.

A research instrument for optical spectroscopy. The central abstraction is ``Method``: every
thing that turns a spectrum into a claim implements one interface, declares machine-readable
assumptions, and is registered with a maturity level. See ``docs/PROJECT_BRIEF.md``.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
