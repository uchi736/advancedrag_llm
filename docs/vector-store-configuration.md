# Vector Store Configuration Guide

This document explains how to configure and use different vector databases with the RAG system.

## Supported Vector Stores

The system now supports two vector store backends:

1. **PGVector** (PostgreSQL with vector extension) - Default
2. **ChromaDB** (Standalone vector database)

## Configuration

### Using PGVector (Default)

PGVector requires a PostgreSQL database with the pgvector extension installed.

Set the following environment variables in your `.env` file:

```env
# Vector Store Configuration
VECTOR_STORE_TYPE=pgvector

# PostgreSQL Configuration (required)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your-password
```

#### Installation Requirements:
- PostgreSQL database with pgvector extension
- `psycopg` or `psycopg2-binary` Python package (already included in requirements.txt)

### Using ChromaDB

ChromaDB can run either as an embedded database or connect to a ChromaDB server.

#### Option 1: Embedded ChromaDB (Local Storage)

```env
# Vector Store Configuration
VECTOR_STORE_TYPE=chromadb

# ChromaDB Local Storage
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

#### Option 2: ChromaDB Server

```env
# Vector Store Configuration
VECTOR_STORE_TYPE=chromadb

# ChromaDB Server Configuration
CHROMA_SERVER_HOST=localhost
CHROMA_SERVER_PORT=8000
CHROMA_SERVER_SSL_ENABLED=false
```

To start a ChromaDB server:
```bash
pip install chromadb
chroma run --host localhost --port 8000
```

## Switching Between Vector Stores

To switch between vector stores:

1. Update the `VECTOR_STORE_TYPE` in your `.env` file
2. Configure the appropriate connection settings
3. Restart your application

**Note:** Data is not automatically migrated between vector stores. You'll need to re-index your documents after switching.

## Architecture

The system uses an adapter pattern to abstract the vector store implementation:

```python
# Vector Store Adapter Interface
class VectorStoreAdapter(ABC):
    def add_documents(documents, ids=None)
    def similarity_search_with_score(query, k=4)
    def delete(ids)
    def get_by_ids(ids)
```

This allows the system to work with different vector stores without changing the core logic.

## Feature Availability by Vector Store

| Feature | PGVector | ChromaDB |
|---------|----------|----------|
| Vector Search | ✅ | ✅ |
| Keyword Search (BM25) | ✅ (Native) | ✅ (BM25) |
| Hybrid Search (Vector + Keyword) | ✅ | ✅ |
| Text-to-SQL | ✅ | ❌ |
| Jargon Dictionary | ✅ (PostgreSQL) | ✅ (ChromaDB Collection) |
| Document Ingestion | ✅ | ✅ |
| Query Expansion | ✅ | ✅ |
| RAG Fusion | ✅ | ✅ |
| Document Chunking | ✅ | ✅ |
| Metadata Filtering | ✅ | ✅ |

**Note:**
- ChromaDB now supports keyword search via BM25 retriever and hybrid search through Reciprocal Rank Fusion
- Jargon Dictionary is supported in ChromaDB using a separate collection
- Text-to-SQL remains unavailable in ChromaDB as it requires SQL database functionality

## When to Use Which Vector Store

### Use PGVector when:
- You already have PostgreSQL infrastructure
- You need SQL capabilities alongside vector search
- You want to keep all data in a single database
- You need full-text search capabilities
- You require Text-to-SQL functionality for data analysis
- You need the jargon dictionary feature for term management
- You want hybrid search (combining vector and keyword search)
- You're running in production with high availability requirements

### Use ChromaDB when:
- PostgreSQL is not available or cannot be configured with pgvector
- You want a simpler, standalone vector database
- You're developing locally or prototyping
- You need a lightweight solution
- Text-to-SQL is not required (only limitation)
- You want full RAG capabilities without PostgreSQL dependency

## Performance Considerations

- **PGVector**: Better for hybrid search (combining vector and SQL queries), requires PostgreSQL overhead
- **ChromaDB**: Faster for pure vector operations, lower resource footprint, but lacks SQL capabilities

## Troubleshooting

### PGVector Issues

If you encounter connection errors:
1. Ensure PostgreSQL is running
2. Verify pgvector extension is installed: `CREATE EXTENSION IF NOT EXISTS vector;`
3. Check connection credentials in `.env`

### ChromaDB Issues

If ChromaDB fails to initialize:
1. For local storage: Ensure write permissions for `CHROMA_PERSIST_DIRECTORY`
2. For server mode: Verify the ChromaDB server is running and accessible
3. Check if the port is not blocked by firewall

## Migration Guide

To migrate existing data from PGVector to ChromaDB:

1. Export documents from PGVector (implement based on your needs)
2. Update `.env` to use ChromaDB
3. Re-index your documents using the ingestion pipeline

Note: Direct migration tools are not currently provided, but the system will work with either backend once configured.

## API Compatibility

Both adapters implement the same interface, ensuring code compatibility:

```python
# Works with both PGVector and ChromaDB
vector_store.add_documents(documents, ids=chunk_ids)
results = vector_store.similarity_search_with_score(query, k=10)
vector_store.delete(ids=chunk_ids)
```

The system automatically handles the differences between implementations internally.