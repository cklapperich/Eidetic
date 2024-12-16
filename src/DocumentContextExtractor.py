from llama_index.core.llms import ChatMessage
from typing import Optional, Dict, List, Tuple, Set
from llama_index.core.llms.llm import LLM
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import Document, Node
from llama_index.core import Settings
from textwrap import dedent

DEFAULT_CONTEXT_PROMPT: str = dedent("""
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Also disambiguate pronouns and key terms in the chunk. Answer only with the succinct context and nothing else.
    """).strip()

DEFAULT_KEY: str = "context"
from llama_index.core.node_parser import TokenTextSplitter
import warnings

class DocumentContextExtractor(BaseExtractor):
    """
    keys: list of keys to add to each node, or a str, if a single key
    prompts: list of prompts, that matches the list of keys, or a single str
    """
    keys: List[str]
    prompts: List[str]
    llm: LLM
    documents: List[Document]
    doc_ids: Set
    max_context_length:int
    
    def _count_document_tokens(self, document):
        text_splitter = TokenTextSplitter(chunk_size=1,chunk_overlap=0)
        tokens = text_splitter.split_text(document.text)
        return len(tokens)
    
    def __init__(self, documents: List[Document], keys=None, prompts=None, llm: LLM = None,
                 num_workers: int = DEFAULT_NUM_WORKERS, max_context_length:int = 128000,
                 node_window=None,
                 ignore_context_length_warning=True, **kwargs):
        
        # Process defaults and values first
        keys = keys or [DEFAULT_KEY]
        prompts = prompts or [DEFAULT_CONTEXT_PROMPT]

        if isinstance(keys, str):
            keys = [keys]
        if isinstance(prompts, str):
            prompts = [prompts]
    
        llm = llm or Settings.llm
        doc_ids = set()

        # Call super().__init__ at the end with all processed values
        super().__init__(
            keys=keys,
            prompts=prompts,
            llm=llm,
            documents=documents,
            num_workers=num_workers,
            doc_ids=doc_ids,
            max_context_length=max_context_length,
            **kwargs
        )

    def set_documents(self, documents: List[Document]):
        for doc in documents:
            self.doc_ids.add(doc.doc_id)
            tokens = self._count_document_tokens(doc)
            if tokens>self.max_context_length:
                print(tokens)
                print(self.max_context_length)
                warnings.warn(f"DocumentExtractorWithCaching Warning: found document with tokens {tokens} greater than max context length ({self.max_context_length}). Filename: {doc.metadata.get('file_name')}")
                if not self.ignore_context_length_warning:
                    warnings.warn("Exiting initializer for DocumentExtractorWithCaching. Set ignore_context_length_warning=True to prevent this and allow initialization to continue, or"
                                  " split your documents into smaller chunks.")
                    return
                
        self.documents = documents
        

    async def _agenerate_node_context(self, node, metadata, document, prompt, key)->Dict:
        
        cached_text = f"<document>{document.text}</document>"

        messages = [
            # ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(
                role="user",
                content=[
                    {
                        "text": cached_text,
                        "block_type": "text",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "text":  f"Here is the chunk we want to situate within the whole document:\n<chunk>{node.text}</chunk>\n{prompt}",
                        "block_type": "text",
                    },
                ],
            ),
        ]
        response = self.llm.chat(
            messages,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        response_text =response.message.blocks[0].text
        metadata[key] = response_text
        return metadata

    async def aextract(self, nodes) -> List[Dict]:
        # Extract node-level summary metadata
        metadata_list: List[Dict] = [{} for _ in nodes]
        # we need to preserve the order of the nodes, but process the nodes uot-of-order
        metadata_map = {node.node_id: metadata_dict for metadata_dict, node in zip(metadata_list, nodes)}

        # make a mapping of doc id: node
        doc_id_to_nodes = {}
        for node in nodes:
            if not (node.source_node and (node.source_node.node_id in self.doc_ids)):
                continue
            parent_id = node.source_node.node_id
            
            if parent_id not in doc_id_to_nodes:
                doc_id_to_nodes[parent_id] = []
            doc_id_to_nodes[parent_id].append(node)
        i = 0
        
        for doc in self.documents: # do this one document at a time for maximum cache efficiency
            node_summaries_jobs = []
            for prompt, key in list(zip(self.prompts, self.keys)):
                for node in doc_id_to_nodes.get(doc.doc_id,[]):
                    i += 1
                    metadata_dict = metadata_map[node.node_id] # get the correct metadata object
                    node_summaries_jobs.append(self._agenerate_node_context(node, metadata_dict, doc, prompt, key))
                    # await self._agenerate_node_context(node, metadata_dict, doc, prompt, key)
                    # node_summaries_jobs.append(self._agenerate_node_context(node, metadata_dict, doc, prompt, key))
            new_metadata = await run_jobs(
                node_summaries_jobs,
                show_progress=self.show_progress,
                workers=self.num_workers,
            )
            print(f"Jobs built. requesting {len(node_summaries_jobs)} nodes with {self.num_workers} workers.")

        print(metadata_list)
        return metadata_list