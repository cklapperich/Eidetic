from enum import Enum
from typing import Optional
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores import QdrantVectorStore
from llama_index.postprocessor import SentenceTransformerRerank
from qdrant_client import QdrantClient

class EmbedderType(Enum):
    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    MINI_LM = "sentence-transformers/all-MiniLM-L6-v2"

class RerankerType(Enum):
    MINI_LM = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    BCE = "cross-encoder/ms-marco-TinyBERT-L-2"

class VectorStoreType(Enum):
    QDRANT = "qdrant"
    IN_MEMORY = "in_memory"

class RetrieverFactory:
    def __init__(
        self,
        embedder_type: str = "BGE_SMALL",
        vector_store_type: str = "QDRANT",
        reranker_type: Optional[str] = "MINI_LM",
        similarity_top_k: int = 2,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ):
        self.embedder_type = EmbedderType[embedder_type]
        self.vector_store_type = VectorStoreType[vector_store_type]
        self.reranker_type = RerankerType[reranker_type] if reranker_type else None
        self.similarity_top_k = similarity_top_k
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        self.retriever = None
        self._setup_retriever()

    def _get_embedder(self):
        return HuggingFaceEmbedding(
            model_name=self.embedder_type.value
        )

    def _get_vector_store(self, embed_model):
        if self.vector_store_type == VectorStoreType.QDRANT:
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            return QdrantVectorStore(
                client=client,
                collection_name="my_collection",
                dim=embed_model.model.get_sentence_embedding_dimension()
            )
        else:
            # Default to in-memory vector store
            return VectorStoreIndex([], embed_model=embed_model).vector_store

    def _get_reranker(self):
        if not self.reranker_type:
            return None
        return SentenceTransformerRerank(
            model_name=self.reranker_type.value,
            top_n=self.similarity_top_k
        )

    def _setup_retriever(self):
        embed_model = self._get_embedder()
        vector_store = self._get_vector_store(embed_model)
        reranker = self._get_reranker()
        
        self.retriever = HybridRetriever(
            vector_store=vector_store,
            embed_model=embed_model,
            similarity_top_k=self.similarity_top_k,
            reranker=reranker
        )

    def set_embedder(self, embedder_type: str):
        """Change the embedding model"""
        self.embedder_type = EmbedderType[embedder_type]
        self._setup_retriever()

    def set_vector_store(self, vector_store_type: str):
        """Change the vector store"""
        self.vector_store_type = VectorStoreType[vector_store_type]
        self._setup_retriever()

    def set_reranker(self, reranker_type: Optional[str] = None):
        """Change or remove the reranker"""
        self.reranker_type = RerankerType[reranker_type] if reranker_type else None
        self._setup_retriever()

    def get_retriever(self) -> HybridRetriever:
        """Get the configured hybrid retriever"""
        return self.retriever
    
if __name__=='__main__':
    # Create with string config
    factory = RetrieverFactory(
        embedder_type="BGE_SMALL",
        vector_store_type="QDRANT",
        reranker_type="MINI_LM",
        similarity_top_k=2
    )

    # Get the configured retriever
    retriever = factory.get_retriever()

    # Swap components easily
    factory.set_embedder("MINI_LM")
    factory.set_reranker("BCE")
    factory.set_vector_store("IN_MEMORY")

    # Remove reranker
    factory.set_reranker(None)