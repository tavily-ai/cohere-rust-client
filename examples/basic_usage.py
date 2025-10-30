#!/usr/bin/env python3
"""Basic usage example for Cohere Rust Client."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from cohere_rust_client import CohereClient


async def main():
    """Demonstrate basic usage of the Cohere Rust Client."""

    # Initialize the client
    print("Initializing Cohere Rust Client...")
    client = CohereClient(
        endpoint_url=os.getenv("COHERE_ENDPOINT_URL", "https://api.cohere.com"),
        model="rerank-v3.5",
        api_key=os.getenv("CO_API_KEY"),
        timeout=30.0,
        max_concurrent_requests=96,
        max_token_limit=50000,  # Lower limit to demonstrate batching
        enable_batching=True
    )

    # Example documents to rerank
    query = "What is machine learning and how does it work?"

    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on enabling computers to learn from data without explicit programming.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "Deep learning is a type of machine learning that uses neural networks with multiple layers to model complex patterns.",
        "The weather today is sunny with a high of 75 degrees Fahrenheit.",
        "Neural networks are computing systems inspired by biological neural networks in animal brains.",
        "JavaScript is primarily used for web development to create interactive websites.",
        "Supervised learning is a machine learning approach where the model learns from labeled training data.",
        "Basketball is a team sport where players score by shooting a ball through a hoop.",
        "Reinforcement learning involves training agents to make decisions by rewarding desired behaviors.",
        "The capital of France is Paris, known for the Eiffel Tower and the Louvre Museum.",
    ]

    print(f"\nQuery: {query}")
    print(f"Number of documents: {len(documents)}\n")

    # Process reranking
    print("Processing reranking request...")
    results, latency = await client.process(
        query=query,
        documents=documents,
        top_k=5  # Return top 5 most relevant documents
    )

    # Display results
    print(f"\n{'='*80}")
    print(f"Reranking completed in {latency:.4f} seconds")
    print(f"{'='*80}\n")

    print(f"Top {len(results)} Results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f} (Original Index: {result.original_index})")
        print(f"   {result.text[:100]}...")
        print()

    # Example with larger payload (demonstrating batching)
    print(f"\n{'='*80}")
    print("Example 2: Large payload with automatic batching")
    print(f"{'='*80}\n")

    # Create a larger set of documents
    large_documents = [
        f"Document {i}: {doc}" * 100  # Make each doc larger
        for i, doc in enumerate(documents * 10)  # Repeat documents
    ]

    print(f"Processing {len(large_documents)} documents...")
    results, latency = await client.process(
        query=query,
        documents=large_documents,
        top_k=10
    )

    print(f"\nCompleted in {latency:.4f} seconds")
    print(f"Returned top {len(results)} results")
    print(f"\nTop 3 results:")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. Score: {result.score:.4f}")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("CO_API_KEY"):
        print("Error: CO_API_KEY environment variable not set")
        print("Please set it in your .env file or environment:")
        print("  export CO_API_KEY='your-api-key'")
        exit(1)

    asyncio.run(main())
