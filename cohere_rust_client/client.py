"""Cohere Reranker client implementation using Baseten Performance Client (Rust-backed).

This client wraps the baseten_performance_client's batch_post() method to provide
high-performance reranking with Cohere's API while benefiting from:
- Rust backend (releases GIL during I/O)
- Efficient connection pooling
- Request hedging support
- High concurrency handling
"""

import asyncio
import json
import logging
import os
import time
from typing import List, Any, Optional, Tuple, Dict

from baseten_performance_client import PerformanceClient
from .base import BaseReranker, RerankResult
from .utils import calculate_reranker_input_tokens, estimate_token_count

# Use standard Python logging
logger = logging.getLogger(__name__)


class CohereClient(BaseReranker):
    """High-performance Cohere Reranker client using Baseten's Rust-backed batch_post().

    This client uses the baseten_performance_client library (written in Rust with pyo3)
    to achieve high throughput reranking while maintaining compatibility with Cohere's API format.

    Performance benefits:
    - GIL-free I/O operations (Rust backend)
    - Efficient connection pooling via reqwest/tokio
    - Request hedging for reduced tail latency
    - Semaphore-based concurrency control
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model: str = "rerank-v3.5-tavily",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_concurrent_requests: int = 96,
        hedge_delay: Optional[float] = None,
        max_token_limit: int = int(os.getenv('COHERE_MAX_TOKEN_LIMIT', 70000)),
        enable_batching: bool = os.getenv('COHERE_ENABLE_BATCHING', 'true').lower() in ('true', '1', 'yes'),
    ):
        """Initialize Cohere Reranker client.

        Args:
            endpoint_url: Cohere endpoint URL (defaults to env var COHERE_ENDPOINT_URL)
            model: Cohere model name (defaults to rerank-v3.5-tavily)
            api_key: Cohere API key (if None, uses CO_API_KEY environment variable)
            timeout: Request timeout in seconds (defaults to 30.0)
            max_concurrent_requests: Maximum concurrent requests (defaults to 96)
            hedge_delay: Optional request hedging delay in seconds (min 0.2s).
                        Sends duplicate requests after delay to improve latency.
            max_token_limit: Maximum token limit per request (defaults to 70000)
            enable_batching: Enable automatic request batching when exceeding token limit (defaults to True)
        """
        super().__init__(timeout=timeout)

        self.endpoint_url = endpoint_url or os.getenv('COHERE_ENDPOINT_URL', 'https://api.cohere.com')
        self.model = model
        self.api_key = api_key or os.getenv('CO_API_KEY')

        if not self.api_key:
            raise ValueError("api_key must be provided or CO_API_KEY environment variable must be set")

        self.max_concurrent_requests = max_concurrent_requests
        self.hedge_delay = hedge_delay
        self.max_candidates = 1000  # Cohere service limit
        self.max_token_limit = max_token_limit
        self.enable_batching = enable_batching

        # Initialize Baseten Performance client (Rust-backed)
        self.client = PerformanceClient(
            base_url=self.endpoint_url,
            api_key=self.api_key
        )

        logger.info(
            f"Cohere Client initialized with model: {self.model}, "
            f"endpoint: {self.endpoint_url}, "
            f"max_concurrent_requests: {self.max_concurrent_requests}, "
            f"hedge_delay: {self.hedge_delay}, "
            f"token_limit: {self.max_token_limit}, batching: {self.enable_batching}"
        )

        # Print endpoint for visibility in logs
        print(f"[COHERE_RUST] Endpoint URL: {self.endpoint_url}")

    async def _split_documents_into_batches(
        self,
        query: str,
        documents: List[str]
    ) -> List[Dict[str, Any]]:
        """Split documents into balanced batches based on token limit.

        Uses a balanced approach: pre-calculates optimal number of batches
        and distributes documents evenly to minimize max batch latency.

        Args:
            query: Search query string
            documents: List of document strings

        Returns:
            List of batch dictionaries, each containing:
                - 'documents': List of document strings in this batch
                - 'original_indices': List of original indices for each document
                - 'token_count': Total token count for this batch
        """
        # Calculate query tokens once (shared across all batches)
        query_tokens = estimate_token_count(query)

        # Pre-calculate token count for each document
        doc_tokens_list = [estimate_token_count(doc) for doc in documents]
        total_doc_tokens = sum(doc_tokens_list)
        total_tokens = query_tokens + total_doc_tokens

        # Calculate optimal number of batches
        # Each batch can hold (max_token_limit - query_tokens) document tokens
        tokens_per_batch = self.max_token_limit - query_tokens
        num_batches = max(1, (total_doc_tokens + tokens_per_batch - 1) // tokens_per_batch)

        # Target tokens per batch for balanced distribution
        target_tokens_per_batch = total_doc_tokens / num_batches

        print(f"[BATCH:BALANCE] Total tokens: {total_tokens}, splitting into {num_batches} batches, target: {target_tokens_per_batch:.0f} tokens/batch")

        # Distribute documents into balanced batches
        batches = []
        current_batch_docs = []
        current_batch_indices = []
        current_batch_tokens = query_tokens

        for idx, (doc, doc_tokens) in enumerate(zip(documents, doc_tokens_list)):
            # Check if we should start a new batch
            # Start new batch if:
            # 1. Current batch would exceed limit, OR
            # 2. We haven't created enough batches yet AND current batch is near target size
            should_start_new_batch = False

            if current_batch_docs:
                potential_total = current_batch_tokens + doc_tokens
                current_batch_doc_tokens = current_batch_tokens - query_tokens

                # Would exceed limit
                if potential_total > self.max_token_limit:
                    should_start_new_batch = True
                # Have more batches to create and current batch is at/above target
                elif len(batches) < num_batches - 1 and current_batch_doc_tokens >= target_tokens_per_batch:
                    should_start_new_batch = True

            if should_start_new_batch:
                # Save current batch
                batches.append({
                    'documents': current_batch_docs,
                    'original_indices': current_batch_indices,
                    'token_count': current_batch_tokens
                })

                # Start new batch with this document
                current_batch_docs = [doc]
                current_batch_indices = [idx]
                current_batch_tokens = query_tokens + doc_tokens
            else:
                # Add document to current batch
                current_batch_docs.append(doc)
                current_batch_indices.append(idx)
                current_batch_tokens += doc_tokens

        # Add final batch if not empty
        if current_batch_docs:
            batches.append({
                'documents': current_batch_docs,
                'original_indices': current_batch_indices,
                'token_count': current_batch_tokens
            })

        # Log batch distribution for debugging
        for i, batch in enumerate(batches):
            print(f"[BATCH:DEBUG] Batch {i+1}/{len(batches)}: {len(batch['documents'])} docs, {batch['token_count']} tokens")

        return batches

    async def _process_batch(
        self,
        query: str,
        batch: Dict[str, Any],
        top_k: Optional[int] = None
    ) -> Tuple[List[RerankResult], float]:
        """Process a single batch of documents.

        Args:
            query: Search query string
            batch: Batch dictionary with 'documents' and 'original_indices'
            top_k: Number of top results to return from this batch

        Returns:
            Tuple of (reranked_results, latency_seconds)
        """
        documents = batch['documents']
        original_indices = batch['original_indices']

        # Construct Cohere API payload
        payload = {
            "query": query,
            "documents": documents,
            "model": self.model,
            "top_n": top_k if top_k else len(documents),
            "return_documents": False
        }

        # Measure JSON serialization time
        serialize_start = time.time()
        payload_json = json.dumps(payload)
        serialize_time = (time.time() - serialize_start) * 1000
        payload_size_kb = len(payload_json) / 1024

        # Use Rust-backed batch_post for high-performance HTTP requests
        start_time = time.time()

        response = await self.client.async_batch_post(
            url_path="/v1/rerank",
            payloads=[payload],
            max_concurrent_requests=self.max_concurrent_requests,
            timeout_s=self.timeout,
            hedge_delay=self.hedge_delay
        )

        rust_call_time = (time.time() - start_time) * 1000
        latency = time.time() - start_time

        # Parse Cohere response format
        deserialize_start = time.time()

        if not response.data:
            raise RuntimeError("Empty response from Cohere API")

        cohere_response = response.data[0]

        # Check for error in response
        if isinstance(cohere_response, dict) and "message" in cohere_response:
            error_msg = cohere_response.get("message", "Unknown error")
            raise RuntimeError(f"Cohere API error: {error_msg}")

        # Build reranked results with original indices
        reranked_results = []
        for result in cohere_response.get("results", []):
            batch_idx = int(result.get("index", 0))
            original_idx = original_indices[batch_idx]

            reranked_results.append(
                RerankResult(
                    text=documents[batch_idx],
                    score=float(result.get("relevance_score", 0.0)),
                    original_index=original_idx,
                )
            )

        deserialize_time = (time.time() - deserialize_start) * 1000

        # Log batch performance breakdown
        print(
            f"[PERF:BATCH] Serialization: {serialize_time:.2f}ms, "
            f"Network+Rust: {rust_call_time:.2f}ms, "
            f"Deserialization: {deserialize_time:.2f}ms, "
            f"Total: {latency*1000:.2f}ms, "
            f"Payload: {payload_size_kb:.2f}KB, "
            f"Docs: {len(documents)}"
        )

        return reranked_results, latency

    async def process(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None
    ) -> Tuple[List[RerankResult], float]:
        """Process a rerank asynchronously using Rust-backed batch_post.

        Args:
            query: Search query string
            documents: List of documents to rerank
            top_k: Number of top results to return (defaults to all documents)

        Returns:
            Tuple of (reranked_results, latency_seconds)
            - reranked_results: List of reranked documents with scores
            - latency_seconds: Time taken for the reranking operation in seconds

        Raises:
            ValueError: If query is empty
            RuntimeError: If the Cohere API request fails
        """
        if not query:
            raise ValueError("Query is required")

        if not documents:
            return [], 0.0

        try:
            # Convert documents to strings and truncate if needed
            doc_strings = [str(doc) for doc in documents]
            truncated_docs = doc_strings[:self.max_candidates]

            if top_k is None:
                top_k = len(truncated_docs)

            # Check if batching is needed
            if self.enable_batching:
                # Calculate total tokens
                total_tokens = calculate_reranker_input_tokens(query, truncated_docs)

                if total_tokens > self.max_token_limit:
                    # Multiple batches needed - split and process in parallel
                    print(f"[BATCH] Splitting {len(truncated_docs)} documents ({total_tokens} tokens) into batches (limit: {self.max_token_limit})...")
                    start_time = time.time()
                    batches = await self._split_documents_into_batches(query, truncated_docs)
                    print(f"[BATCH] Created {len(batches)} batches in {time.time() - start_time:.2f} seconds")

                    # Process all batches in parallel using asyncio.gather
                    batch_tasks = [
                        self._process_batch(query, batch, top_k=None)
                        for batch in batches
                    ]

                    # Wait for all batches to complete
                    batch_start = time.time()
                    batch_results = await asyncio.gather(*batch_tasks)
                    batch_total_time = (time.time() - batch_start) * 1000

                    # Merge results from all batches
                    all_results = []
                    max_latency = 0.0
                    min_latency = float('inf')
                    sum_latency = 0.0

                    for results, latency in batch_results:
                        all_results.extend(results)
                        max_latency = max(max_latency, latency)
                        min_latency = min(min_latency, latency)
                        sum_latency += latency

                    avg_batch_latency = sum_latency / len(batch_results)

                    # Sort all results by score (descending)
                    all_results.sort(key=lambda x: x.score, reverse=True)

                    # Apply top_k globally
                    final_results = all_results[:top_k]

                    print(
                        f"[BATCH] Merged {len(all_results)} results, returning top {len(final_results)} | "
                        f"Parallel time: {batch_total_time:.2f}ms, "
                        f"Batch latencies - Min: {min_latency*1000:.2f}ms, "
                        f"Avg: {avg_batch_latency*1000:.2f}ms, "
                        f"Max: {max_latency*1000:.2f}ms"
                    )

                    return final_results, max_latency

            # Single request (no batching needed or batching disabled)
            # Construct Cohere API payload
            payload = {
                "query": query,
                "documents": truncated_docs,
                "model": self.model,
                "top_n": top_k,
                "return_documents": False
            }

            # Measure JSON serialization time
            serialize_start = time.time()
            payload_json = json.dumps(payload)
            serialize_time = (time.time() - serialize_start) * 1000
            payload_size_kb = len(payload_json) / 1024

            # Use Rust-backed batch_post for high-performance HTTP requests
            start_time = time.time()

            response = await self.client.async_batch_post(
                url_path="/v1/rerank",
                payloads=[payload],
                max_concurrent_requests=self.max_concurrent_requests,
                timeout_s=self.timeout,
                hedge_delay=self.hedge_delay
            )

            rust_call_time = (time.time() - start_time) * 1000
            latency = time.time() - start_time

            # Parse Cohere response format
            # response.data is a list of response objects (one per payload)
            deserialize_start = time.time()

            if not response.data:
                raise RuntimeError("Empty response from Cohere API")

            cohere_response = response.data[0]

            # Check for error in response
            if isinstance(cohere_response, dict) and "message" in cohere_response:
                error_msg = cohere_response.get("message", "Unknown error")
                raise RuntimeError(f"Cohere API error: {error_msg}")

            deserialize_time = (time.time() - deserialize_start) * 1000

            # Build reranked results from Cohere response
            reranked_results = []
            for result in cohere_response.get("results", []):
                original_idx = int(result.get("index", 0))

                reranked_results.append(
                    RerankResult(
                        text=truncated_docs[original_idx],
                        score=float(result.get("relevance_score", 0.0)),
                        original_index=original_idx,
                    )
                )

            # Log detailed performance breakdown
            print(
                f"[PERF] Serialization: {serialize_time:.2f}ms, "
                f"Network+Rust: {rust_call_time:.2f}ms, "
                f"Deserialization: {deserialize_time:.2f}ms, "
                f"Total: {latency*1000:.2f}ms, "
                f"Payload: {payload_size_kb:.2f}KB, "
                f"Docs: {len(truncated_docs)}"
            )

            logger.debug(
                f"Reranked {len(truncated_docs)} documents to top {len(reranked_results)} "
                f"in {latency:.4f}s"
            )

            return reranked_results, latency

        except Exception as e:
            logger.error(f"Cohere reranking failed: {str(e)}")
            raise RuntimeError(f"Cohere reranking failed: {str(e)}")

    def __repr__(self) -> str:
        """String representation of the client."""
        return (
            f"CohereClient(model={self.model}, "
            f"endpoint={self.endpoint_url}, "
            f"max_concurrent_requests={self.max_concurrent_requests}, "
            f"hedge_delay={self.hedge_delay})"
        )
