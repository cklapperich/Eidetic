from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import Document, Node, TransformComponent
from llama_index.core.settings import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors import BaseExtractor
from datetime import datetime
from pathlib import Path

@dataclass
class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector search and BM25 with optional reranking and contextual embeddings."""
    
    index: VectorStoreIndex = None
    reranker: Optional[BaseRetriever] = None

    similarity_top_k: int = 2

    cached_contextual_nodes: Dict[str, Node] = field(default_factory=dict)

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


class DocumentContextExtractorWithAnthropicCaching(BaseExtractor):
    """
    keys: list of keys to add to each node, or a str, if a single key
    prompts: list of prompts, that matches the list of keys, or a single str
    """
    keys: List[str] | str
    prompts: List[str] | str
    llm: LLM
    system_prompt: str
    documents: dict

    DEFAULT_CONTEXT_PROMPT = """\
    Here is the chunk we want to situate within the whole document
    <chunk>
    {CHUNK_CONTENT}
    </chunk>
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
    
    DEFAULT_KEY = "context"

    def __init__(self, documents:List[Document], keys=None, prompts=None,  llm:LLM = None, num_workers:int=DEFAULT_NUM_WORKERS, **kwargs):

        self.keys = keys or [self.DEFAULT_KEY]
        self.prompts = prompts or [self.DEFAULT_CONTEXT_PROMPT]

        if isinstance(self.keys, str):
            self.keys = [self.keys]
        if isinstance(self.prompts, str):
            self.prompts = [self.prompts]
    
        self.llm = llm or Settings.llm
        self.num_workers = num_workers
        self.documents = {}

        for doc in documents:
            self.documents[doc.doc_id] = doc
        
        # TODO: do we need this?? probably not
        # super().__init__(
        #     **kwargs,
        # )

    async def _agenerate_node_context(self, node, metadata, document, prompt, key)->Dict:
        messages = [
            # ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(
                role="user",
                content=[
                    {
                        "text": self.prompt_document.format(
                            WHOLE_DOCUMENT=document.text
                        ),
                        "type": "text",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "text": prompt.format(CHUNK_CONTENT=node.text),
                        "type": "text",
                    },
                ],
            ),
        ]

        metadata[key] = str(
            self.context_llm.chat(
                messages,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )
        )

        return metadata

    async def aextract(self, nodes) -> List[Dict]:
        # Extract node-level summary metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        # we need to preserve the order of the nodes, but process the nodes out-of-order
        metadata_map = {node.node_id: metadata_dict for metadata_dict, node in zip(metadata_list, nodes)}

        # make a mapping of doc id: node
        doc_id_to_nodes = {}
        for node in nodes:
            parent_id = node.source_node.node_id
            if parent_id not in doc_id_to_nodes:
                doc_id_to_nodes[parent_id] = []
            doc_id_to_nodes[parent_id].append(node)

        node_summaries_jobs = []
        for doc in self.documents: # do this one document at a time for maximum cache efficiency
            for prompt, key in list(zip(self.prompts, self.keys)):
                for node in doc_id_to_nodes.get(doc.doc_id,[]):
                    metadata_dict = metadata_map[node.node_id] # get the correct metadata object
                    node_summaries_jobs.append(self._agenerate_node_summary(node, metadata_dict, doc, prompt, key))

            # each batch of jobs is 1 single document
            new_metadata = await run_jobs(
                node_summaries_jobs,
                show_progress=self.show_progress,
                workers=self.num_workers,
            )
            # no need to do anything with new_metadata, the list will be in the wrong order, and we're already modifying the metadata list in-place

        return metadata_list
    
if __name__=='__main__':
    # Create components
    data_path = Path('./data')
    print(data_path)
    assert(data_path.exists())
    
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Contextualizer = DocumentContextExtractorWithAnthropicCaching()
    
    transformations = [Contextualizer]

    vector_store = SimpleVectorStore()
    
    index = VectorStoreIndex(
        [], #  empty list of nodes
        embed_model=embed_model,
        vector_store=vector_store,
        transformations=transformations
    )

    reranker = SentenceTransformerRerank(
        top_n=2
    )

    # Create hybrid retriever
    retriever = HybridRetriever(
        VectorStoreIndex=VectorStoreIndex,
        embed_model=embed_model,
        similarity_top_k=4,
        reranker=reranker
    )

    retriever.load_directory(data_path)

    # per-index
    index = VectorStoreIndex.from_documents(
        documents, transformations=transformations
    )