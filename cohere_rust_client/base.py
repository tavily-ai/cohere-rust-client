"""Base classes for reranker clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any, Optional, Tuple


@dataclass
class RerankResult:
    """Standard result format for reranked documents.

    Attributes:
        text: The document text
        score: Relevance score (higher is more relevant)
        original_index: Index of this document in the original input list
    """
    text: str
    score: float
    original_index: int


class BaseReranker(ABC):
    """Abstract base class for reranker implementations."""

    def __init__(self, timeout: float = 30.0):
        """Initialize base reranker.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    @abstractmethod
    async def process(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> Tuple[List[RerankResult], float]:
        """Process a rerank asynchronously.

        Args:
            query: Search query string
            documents: List of documents to rerank
            top_k: Number of top results to return (defaults to all documents)

        Returns:
            Tuple of (reranked_results, latency_seconds)
            - reranked_results: List of reranked documents with scores
            - latency_seconds: Time taken for the reranking operation in seconds
        """
        pass
