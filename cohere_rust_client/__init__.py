"""
Cohere Rust Client - High-performance reranker with intelligent batching

This package provides a high-performance Python client for Cohere's reranking API
with Rust-backed HTTP and intelligent batching capabilities.

Features:
- Rust-backed HTTP via baseten-performance-client (GIL-free I/O)
- Intelligent balanced batching for large requests
- Request hedging support for reduced tail latency
- Performance instrumentation and detailed logging
- Configurable token limits and concurrency
"""

__version__ = "0.1.0"
__author__ = "Tavily"

from .client import CohereClient
from .base import RerankResult

__all__ = ["CohereClient", "RerankResult"]
