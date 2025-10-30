# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-30

### Added
- Initial release of Cohere Rust Client
- `CohereClient` with Rust-backed HTTP via baseten-performance-client
- Balanced batch splitting algorithm for large requests
- Automatic token-based request batching
- Request hedging support for reduced tail latency
- Performance instrumentation with detailed timing breakdowns
- `RerankResult` dataclass for standardized output
- `BaseReranker` abstract base class
- Comprehensive README with usage examples
- Example scripts demonstrating basic usage
- GitHub Actions workflow for CodeArtifact deployment

### Performance Features
- GIL-free I/O operations via Rust backend
- Efficient connection pooling
- Balanced batch distribution (reduces P95 latency by ~50% for imbalanced cases)
- Configurable concurrency and token limits

### Documentation
- Complete API reference
- Configuration guide with environment variables
- Advanced usage examples (batching, hedging)
- Performance monitoring guide
