<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Copilot Instructions for LLM-Powered Query-Retrieval System

This is a FastAPI-based intelligent document Q&A system using Retrieval-Augmented Generation (RAG).

## Project Context

- **Framework**: FastAPI for REST API development
- **ML/AI Stack**: sentence-transformers, FAISS, OpenAI/Claude integration
- **Document Processing**: PyMuPDF for PDF parsing and text extraction
- **Vector Search**: FAISS for efficient similarity search
- **Authentication**: Bearer token-based security

## Code Style Guidelines

1. **Async/Await**: Use async/await for I/O operations (file downloads, API calls)
2. **Type Hints**: Always include comprehensive type hints for function parameters and return values
3. **Error Handling**: Implement comprehensive error handling with specific HTTP exceptions
4. **Logging**: Use structured logging with appropriate log levels
5. **Validation**: Use Pydantic models for request/response validation
6. **Documentation**: Include detailed docstrings for all classes and methods

## Architecture Patterns

- **Separation of Concerns**: Each class handles a specific responsibility (DocumentProcessor, TextChunker, VectorStore, LLMService)
- **Dependency Injection**: Use FastAPI's dependency injection for authentication and shared resources
- **Resource Management**: Proper cleanup of temporary files and connections
- **Configuration**: Use constants for configurable parameters

## Security Considerations

- Always validate input data (URLs, file sizes, content types)
- Implement proper authentication and authorization
- Handle sensitive data (API keys) securely
- Validate file uploads and prevent directory traversal

## Performance Best Practices

- Use async operations for concurrent processing
- Implement efficient batching for embeddings
- Optimize memory usage for large documents
- Use appropriate indexing strategies for vector search

## Testing Considerations

- Mock external API calls (OpenAI, document downloads)
- Test edge cases (empty documents, invalid URLs, large files)
- Validate response formats and error conditions
- Performance testing for concurrent requests
