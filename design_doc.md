# Eidetic - Claude Memory extender

Eidetic is a more than just an MCP server for claude. 

It's:
1. an entire SOTA RAG server 
2. an MCP server built on top of that

1st we need to built an awesome configurable SOTA rag server with an API. 
then we can build a MCP server on top of it.

## Motivation

I dont like reading files via filesystem and I think the graph memory plugin is bad.

I want something vector based for 2 purposes:

1. RAG chat over large personal document datasets

2. making a better version of chatgpt "Memory" feature. 
a. have a 'memory' text document
b. keep updated - no need to chunk it, , but we can add context
c. query it automatically with EVERY chat? or at least let claude do 'remember' as a command, which runs the
search algo on the memory document. 
System prompts can encourage it to use memory frequently.
 
3. Combine with Filesystem MCP server. Allow claude to read the entire file when it finds interesting things.
need some way to associate chunks with filenames.

4. need a realy good system prompt to tie this all together, like claude's artifacts prompt

5. add a custom contextualizer prompt feature the user can edit

Inspired by: https://www.anthropic.com/news/contextual-retrieval
and reddit posts asking for an MCP pinecone plugin

support for pinecone may come later, this is QDrant for now
add more DB support later

## Vector Database Selection

Probably QDrant because

- **Qdrant**
  - Rust-based, excellent performance
  - Built-in hybrid search with BM25-like scoring
  - Local deployment via file-based storage
  - Strong Python SDK
  - Active development
  - both cloud and local
  - local is free

- **Alternative Options**
  - Weaviate: Supports hybrid search but requires Docker
  - Milvus: Good hybrid search but deployment is more complex

## Llama Index
Llama Index provides an ideal framework for implementing vector storage due to:

**Core Features**:
  - Data ingestion and structuring
  - Vector store management
  - Query routing
  - Built-in evaluation tools
  - Streaming and async support

3. **Vector Store Support**:
```python
from llama_index.vector_stores import (
    ChromaVectorStore,
    QdrantVectorStore, 
    WeaviateVectorStore,
    FaissVectorStore,
    PineconeVectorStore
)
```

4. **Basic Implementation**:
```python
from llama_index.vector_stores import VectorStoreQuery
from llama_index.schema import TextNode

# Initialize store
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("your_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Add documents
nodes = [
    TextNode(text="some text", id_="node1"),
    TextNode(text="other text", id_="node2")
]
vector_store.add(nodes)

# Query
query = VectorStoreQuery(query_embedding=[0.1, 0.2, ...])
results = vector_store.query(query)
```

## Contextual Retrieval Implementation

https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb

## fastmcp

[python library this MCP server will be built with](https://github.com/jlowin/fastmcp/blob/main/README.md)

### 1. Document Processing Pipeline
- Split documents into manageable chunks (few hundred tokens)
- Generate contextual descriptions using Claude:
  ```
  <document> 
  {{WHOLE_DOCUMENT}} 
  </document> 
  Here is the chunk we want to situate within the whole document 
  <chunk> 
  {{CHUNK_CONTENT}} 
  </chunk> 
  Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
  ```
- Prepend context to each chunk before indexing

### 2. Dual Indexing System
- Create and store semantic embeddings of contextualized chunks in vector DB
- Build BM25/TF-IDF index on raw text of contextualized chunks
  - BM25 works on actual text, not vectors
  - Looks at term frequency, document length, and inverse document frequency
  - Complements vector search by finding exact matches
- Need two types of storage:
  1. Vector DB for embeddings (semantic search)
  2. Text index for BM25 (lexical search)
- Ensure both indexes support efficient querying
- Note: Both indexes work with the same contextualized chunks, just indexed differently

### 3. Retrieval Pipeline
1. Query Processing:
   - Search vector DB for semantic similarity matches
   - Search BM25 index for exact matches
   - Combine and deduplicate results
2. Optional Reranking:
   - Use reranker model (e.g., Cohere) to improve relevance
   - Score and filter top results
3. Return top-K chunks (recommended K=20)

### 4. System Configuration Parameters
- Chunk size and overlap settings
- Embedding model selection (recommended: Gemini or Voyage)
- Reranking configuration
- Number of chunks to return (K value)
- Contextualization prompt template