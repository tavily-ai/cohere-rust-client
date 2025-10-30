# Cohere Rust Client

High-performance Python client for Cohere's reranking API with Rust-backed HTTP and intelligent batching.

## Features

- 🚀 **Rust-backed HTTP** - GIL-free I/O via baseten-performance-client for true parallelism
- 📊 **Intelligent Batching** - Automatically splits large requests with balanced token distribution
- ⚡ **Request Hedging** - Optional duplicate requests for reduced tail latency
- 📈 **Performance Instrumentation** - Detailed timing breakdowns for debugging
- 🔧 **Configurable** - Token limits, concurrency, timeouts all customizable
- 🎯 **Production-ready** - Battle-tested in high-throughput environments

## Installation

```bash
pip install cohere-rust-client
```

## Quick Start

```python
import asyncio
from cohere_rust_client import CohereClient

async def main():
    # Initialize client
    client = CohereClient(
        endpoint_url="https://api.cohere.com",
        model="rerank-v3.5",
        api_key="your-api-key",
        max_token_limit=50000,
        enable_batching=True
    )

    # Rerank documents
    results, latency = await client.process(
        query="What is machine learning?",
        documents=[
            "Machine learning is a subset of artificial intelligence...",
            "Python is a programming language...",
            "Deep learning uses neural networks...",
        ],
        top_k=10
    )

    # Process results
    for result in results:
        print(f"Score: {result.score:.4f}")
        print(f"Text: {result.text[:100]}...")
        print(f"Original Index: {result.original_index}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### Client Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `endpoint_url` | Cohere API endpoint | `COHERE_ENDPOINT_URL` env or `https://api.cohere.com` |
| `model` | Model name | `rerank-v3.5-tavily` |
| `api_key` | API key | `CO_API_KEY` env variable |
| `timeout` | Request timeout (seconds) | `30.0` |
| `max_concurrent_requests` | Max parallel requests | `96` |
| `hedge_delay` | Hedging delay (seconds) | `None` (disabled) |
| `max_token_limit` | Token limit per request | `70000` |
| `enable_batching` | Enable auto-batching | `True` |

### Environment Variables

```bash
# Required
export CO_API_KEY="your-cohere-api-key"

# Optional
export COHERE_ENDPOINT_URL="https://api.cohere.com"
export COHERE_MAX_TOKEN_LIMIT="50000"
export COHERE_ENABLE_BATCHING="true"
```

## Advanced Usage

### Automatic Batching

When your request exceeds the token limit, the client automatically:
1. Calculates optimal number of batches
2. Distributes documents evenly across batches
3. Processes batches in parallel
4. Merges and ranks results

```python
# Large request with 1000+ documents
results, latency = await client.process(
    query="search query",
    documents=large_document_list,  # e.g., 1000 documents
    top_k=50
)

# Client automatically:
# - Splits into balanced batches (e.g., 2 batches of ~25K tokens each)
# - Processes batches in parallel
# - Returns top 50 globally ranked results
```

### Request Hedging

Reduce tail latency by sending duplicate requests after a delay:

```python
client = CohereClient(
    api_key="your-key",
    hedge_delay=0.5  # Send duplicate request after 500ms
)

# If the first request takes > 500ms, a duplicate is sent
# First response to complete is used, others are cancelled
```

### Performance Monitoring

The client logs detailed performance metrics:

```
[PERF] Serialization: 0.15ms, Network+Rust: 245.32ms, Deserialization: 0.43ms, Total: 245.90ms, Payload: 123.45KB, Docs: 500
```

For batched requests:
```
[BATCH:BALANCE] Total tokens: 61529, splitting into 2 batches, target: 30764 tokens/batch
[BATCH:DEBUG] Batch 1/2: 435 docs, 46205 tokens
[BATCH:DEBUG] Batch 2/2: 87 docs, 45245 tokens
[BATCH] Merged 522 results, returning top 50 | Parallel time: 156.23ms, Batch latencies - Min: 148.12ms, Avg: 152.18ms, Max: 156.23ms
```

## Performance

### Rust Backend Benefits

- **GIL-free I/O**: True parallelism in Python via Rust backend
- **Connection pooling**: Efficient HTTP connection reuse
- **Low overhead**: Minimal serialization/deserialization cost (~0.5ms total)

### Balanced Batching

Traditional sequential batching can create severe imbalances:
- ❌ Batch 1: 50K tokens (99% capacity)
- ❌ Batch 2: 2.6K tokens (5% capacity)
- 😞 Result: P95 latency = max(batch1_latency, batch2_latency)

Our balanced batching ensures even distribution:
- ✅ Batch 1: 46K tokens (92% capacity)
- ✅ Batch 2: 45K tokens (90% capacity)
- 🎉 Result: ~50% reduction in P95 latency for imbalanced cases

## API Reference

### CohereClient

```python
class CohereClient(BaseReranker):
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model: str = "rerank-v3.5-tavily",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_concurrent_requests: int = 96,
        hedge_delay: Optional[float] = None,
        max_token_limit: int = 70000,
        enable_batching: bool = True,
    )
```

### RerankResult

```python
@dataclass
class RerankResult:
    text: str               # Document text
    score: float            # Relevance score (0.0 to 1.0)
    original_index: int     # Index in original input list
```

### process()

```python
async def process(
    self,
    query: str,
    documents: List[Any],
    top_k: Optional[int] = None
) -> Tuple[List[RerankResult], float]:
    """
    Process reranking request.

    Args:
        query: Search query string
        documents: List of documents to rerank
        top_k: Number of top results to return (default: all)

    Returns:
        Tuple of (reranked_results, latency_seconds)
    """
```

## Error Handling

```python
try:
    results, latency = await client.process(
        query="search query",
        documents=documents,
        top_k=10
    )
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"API error: {e}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: https://github.com/tavily-ai/cohere-rust-client/issues
- Email: dev@tavily.com
