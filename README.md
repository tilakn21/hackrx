# LLM-Powered Intelligent Query-Retrieval System

A production-grade FastAPI service that provides intelligent document Q&A using Retrieval-Augmented Generation (RAG) for PDF documents.

## Features

- **PDF Processing**: Download and parse PDFs from blob storage URLs
- **Semantic Chunking**: Split content into meaningful chunks (paragraphs or 500 tokens max)
- **Vector Embeddings**: Use sentence-transformers (all-MiniLM-L6-v2) for text embeddings
- **Fast Retrieval**: FAISS-powered vector search for relevant context
- **LLM Integration**: Claude/OpenAI integration for answer generation
- **Production Ready**: Comprehensive error handling, logging, and validation

## API Endpoints

### Main Query Endpoint
`POST /api/v1/hackrx/run`

**Headers:**
```
Authorization: Bearer 4e604cba5381f75493ad0742118a94a430a6ac0f7efd3ff7e86ede5705dc5487
```

**Request Body:**
```json
{
  "documents": "<PDF blob URL>",
  "questions": [
    "What is the waiting period for pre-existing diseases?",
    "Does the policy cover maternity expenses?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "There is a waiting period of 36 months for PED.",
    "Yes, the policy covers maternity after 24 months of continuous coverage."
  ]
}
```

### Additional Endpoints

- `GET /health` - Health check
- `GET /api/v1/stats` - System statistics (requires auth)

## Installation & Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   python main.py
   ```
   
   Or with uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - OpenAPI spec: http://localhost:8000/openapi.json

## Architecture

### 1. Document Processing
- Downloads PDFs from blob storage URLs
- Extracts text using PyMuPDF
- Validates PDF size (max 50MB)

### 2. Text Chunking
- Semantic chunking based on paragraphs
- Token-based splitting (max 500 tokens per chunk)
- Preserves context and meaning

### 3. Vector Search
- Creates embeddings using sentence-transformers
- FAISS index for fast similarity search
- Cosine similarity with configurable threshold

### 4. Answer Generation
- Retrieves top-K relevant chunks
- Provides context to LLM (Claude/OpenAI)
- Returns structured answers with supporting evidence

## Configuration

Key parameters in `main.py`:

- `MAX_CHUNK_SIZE = 500` - Maximum tokens per chunk
- `SIMILARITY_THRESHOLD = 0.7` - Minimum similarity for relevance
- `TOP_K_CHUNKS = 5` - Number of chunks to retrieve
- `MAX_PDF_SIZE_MB = 50` - Maximum PDF file size

## Security

- Bearer token authentication required
- Input validation and sanitization
- Comprehensive error handling
- Request size limits

## Performance Optimizations

- Async/await for concurrent operations
- Efficient vector indexing with FAISS
- Batch processing for embeddings
- Memory-efficient PDF processing

## Error Handling

- Invalid PDF URLs
- Network connectivity issues
- PDF parsing errors
- Empty or corrupted documents
- Authentication failures
- Rate limiting considerations

## Testing

The system includes a simulation mode for LLM responses when API keys are not available. For production use, replace the `_simulate_llm_response` method with actual OpenAI/Claude API calls.

## Production Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

# Test the application
python test_app.py
```

### Render Deployment (Free Tier)

This application is optimized for deployment on Render's free tier. See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

**Quick Deployment:**
1. Push your code to GitHub
2. Connect GitHub repo to Render
3. Configure environment variables
4. Deploy automatically

**Render Configuration:**
- **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
- **Start Command**: `python main.py`
- **Health Check**: `/health`
- **Port**: 10000 (automatically configured)

### Environment Variables
```bash
PYTHONUNBUFFERED=1
PORT=10000
PYTHONPATH=.
```

For production deployment:

1. Set up proper environment variables for API keys
2. Configure logging and monitoring
3. Set up load balancing and scaling
4. Implement caching for frequently accessed documents
5. Add rate limiting and request queuing
6. Set up health checks and metrics

## Testing

Run the test script to verify everything works:
```bash
python test_app.py
```

## Dependencies

- **FastAPI**: Web framework
- **sentence-transformers**: Text embeddings
- **FAISS**: Vector similarity search
- **PyMuPDF**: PDF text extraction
- **tiktoken**: Token counting
- **aiohttp**: Async HTTP client
- **OpenAI**: LLM integration (optional)

## Files Structure
```
├── main.py              # Main FastAPI application
├── config.py            # Configuration and constants
├── requirements.txt     # Python dependencies
├── render.yaml         # Render deployment configuration
├── Procfile            # Process configuration
├── runtime.txt         # Python version specification
├── Dockerfile          # Docker configuration (optional)
├── test_app.py         # Local testing script
├── DEPLOYMENT.md       # Detailed deployment guide
└── README.md           # This file
```

## License

This project is for internal use only.
