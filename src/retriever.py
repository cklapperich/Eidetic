from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from llama_index.core.schema import NodeWithScore, QueryBundle

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader

from llama_index.core.vector_stores import SimpleVectorStore

from llama_index.core.embeddings import BaseEmbedding

from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import Document, Node

import copy
from datetime import datetime
from pathlib import Path

@dataclass
class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector search and BM25 with optional reranking and contextual embeddings."""
    
    vector_store: SimpleVectorStore  = None
    embed_model: BaseEmbedding = None
    context_llm: Optional[object] = None
    reranker: Optional[BaseRetriever] = None

    similarity_top_k: int = 2
    documents: dict[str, Document] = field(default_factory=dict)
    nodes: List[Node] = field(default_factory=list)
    data_paths: List[Path] = field(default_factory=list)

    prompt_document = """\
<document>
{WHOLE_DOCUMENT} 
</document>
"""

    prompt_chunk = """\
Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""

    context_system_prompt: str = "You are a helpful AI Assistant."
    cached_contextual_nodes: Dict[str, Node] = field(default_factory=dict)

    def add_document(self, document):
        self.documents[document.doc_id] = document
        new_nodes = self._create_contextual_nodes_for_document(
            self.prompt_document, document, self.cached_contextual_nodes
        )

        # get embeddings for the nodes
        # simple vector store has no 'add documents'
        self.vector_store.add(new_nodes)

        # add to the mongodb database - or just the list of nodes for now
        self.nodes.extend(new_nodes)
        

    def load_directory(self, data_path:Path):
        # document metadata keys:
        # dict_keys(['file_path', 'file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date'])

        if not isinstance(data_path, Path):
            data_path = Path(data_path)

        if not data_path.exists():
            raise ValueError(f"Database path {data_path} does not exist.")
        
        reader = SimpleDirectoryReader(data_path, filename_as_id=True)
        new_documents = reader.load_data()
        self.data_paths.append(data_path)
        for doc in new_documents:
            result = self.add_document(doc)

    def _create_contextual_nodes_for_document(self, prompt_document, document: Document, nodes: List[Node]) -> List[Node]:
        if not self.context_llm or not self.context_prompt_template:
            return nodes

        nodes_modified = []
        for node in nodes:
            new_node = copy.deepcopy(node)
            messages = [
                ChatMessage(role="system", content="You are helpful AI Assistant."),
                ChatMessage(
                    role="user",
                    content=[
                        {
                            "text": prompt_document.format(
                                WHOLE_DOCUMENT=document.text
                            ),
                            "type": "text",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "text": self.prompt_chunk.format(CHUNK_CONTENT=node.text),
                            "type": "text",
                        },
                    ],
                ),
            ]

            # by default, filename is the document id
            # new_node.metadata["filename"] = document.id
            new_node.metadata["context"] = str(
                self.context_llm.chat(
                    messages,
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                )
            )
            nodes_modified.append(new_node)

        return nodes_modified

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        # TODO: Should we recreate the bm25 retriever and vector retriever each time? or leave them static as a part of the class?

        # Create vector index and retriever
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
        
        # If we have context generation enabled, create contextual nodes
        if self.context_llm and self.context_prompt_template:
            nodes = self._create_contextual_nodes(
                list(vector_index.docstore.docs.values())
            )
            # Recreate vector index with contextual nodes
            vector_index = VectorStoreIndex(nodes, embed_model=self.embed_model)
            
        vector_retriever = vector_index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=vector_index.docstore.docs.values(),
            similarity_top_k=self.similarity_top_k
        )
        
        # Get results from both retrievers
        vector_nodes = vector_retriever.retrieve(query_bundle)
        bm25_nodes = bm25_retriever.retrieve(query_bundle)
        
        # Combine results
        # TODO: Reciprocal rank fusion
        """
        # Combine results
        chunk_ids = list(set(ranked_chunk_ids + ranked_bm25_chunk_ids))
        chunk_id_to_score = {}

        # Initial scoring with weights
        for chunk_id in chunk_ids:
            score = 0
            if chunk_id in ranked_chunk_ids:
                index = ranked_chunk_ids.index(chunk_id)
                score += semantic_weight * (1 / (index + 1))  # Weighted 1/n scoring for semantic
            if chunk_id in ranked_bm25_chunk_ids:
                index = ranked_bm25_chunk_ids.index(chunk_id)
                score += bm25_weight * (1 / (index + 1))  # Weighted 1/n scoring for BM25
            chunk_id_to_score[chunk_id] = score

        # Sort chunk IDs by their scores in descending order
        sorted_chunk_ids = sorted(
            chunk_id_to_score.keys(), key=lambda x: (chunk_id_to_score[x], x[0], x[1]), reverse=True
        )

        # Assign new scores based on the sorted order
        for index, chunk_id in enumerate(sorted_chunk_ids):
        chunk_id_to_score[chunk_id] = 1 / (index + 1)
        """
        all_nodes = vector_nodes + bm25_nodes
        
        # Rerank if reranker provided
        if self.reranker:
            all_nodes = self.reranker.postprocess_nodes(all_nodes, query_bundle)
            
        return all_nodes[:self.similarity_top_k]

if __name__=='__main__':
    # Create components
    data_path = Path('./data')
    print(data_path)
    assert(data_path.exists())
    
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = SimpleVectorStore()

    reranker = SentenceTransformerRerank(
        top_n=2
    )

    # Create hybrid retriever
    retriever = HybridRetriever(
        vector_store=vector_store,
        embed_model=embed_model,
        similarity_top_k=4,
        reranker=reranker
    )

    retriever.load_directory(data_path)