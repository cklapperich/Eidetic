from typing import Any, Callable, List, Optional, Sequence, Dict
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.node_parser import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core import Settings
from textwrap import dedent
import importlib
import logging
from llama_index.core.node_parser import TokenTextSplitter

DEFAULT_CONTEXT_PROMPT: str = dedent("""
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Also disambiguate pronouns and key terms in the chunk. Answer only with the succinct context and nothing else.
    """).strip()

DEFAULT_KEY: str = "context"

class DocumentContextNodeParser(NodeParser):
    """Adds document context to each node by analyzing parent documents."""

    def __init__(
        self,
        llm: Optional[LLM] = None,
        keys: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
        num_workers: int = DEFAULT_NUM_WORKERS,
        max_context_length: int = 128000,
        max_contextual_tokens: int = 512,
        oversized_document_strategy: str = "truncate_first",
        **kwargs: Any,
    ):
        """Initialize params."""
        # Check tiktoken requirement - because tokentextsplitter is required which requires tiktoken
        if not importlib.util.find_spec("tiktoken"):
            raise ValueError("TikToken is required for DocumentContextNodeParser. Please install tiktoken.")

        # Process defaults
        keys = keys or [DEFAULT_KEY]
        prompts = prompts or [DEFAULT_CONTEXT_PROMPT]
        
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(prompts, str):
            prompts = [prompts]

        llm = llm or Settings.llm

        super().__init__(
            num_workers=num_workers,
            **kwargs
        )

        self.llm = llm
        self.keys = keys
        self.prompts = prompts
        self.max_context_length = max_context_length
        self.max_contextual_tokens = max_contextual_tokens
        self.oversized_document_strategy = oversized_document_strategy

    @staticmethod
    def _truncate_text(text: str, max_token_count: int, how='first') -> str:
        """Truncate text to specified token count."""
        text_splitter = TokenTextSplitter(chunk_size=max_token_count, chunk_overlap=0)
        chunks = text_splitter.split_text(text)
        if how == 'first':
            text = chunks[0]
        elif how == 'last':
            text = chunks[-1]
        else:
            raise ValueError("Invalid truncation method. Must be 'first' or 'last'.")
        
        return text if text else ""
    
    @staticmethod
    def _count_tokens(text: str) -> int:
        """Count tokens in text."""
        text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
        tokens = text_splitter.split_text(text)
        return len(tokens)

    async def _agenerate_node_context(self, node: BaseNode, doc_content: str, prompt: str) -> str:
        """Generate context for a node using parent document content."""
        cached_text = f"<document>{doc_content}</document>"

        messages = [
            ChatMessage(
                role="user",
                content=[
                    {
                        "text": cached_text,
                        "block_type": "text",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "text": f"Here is the chunk we want to situate within the whole document:\n<chunk>{node.get_content()}</chunk>\n{prompt}",
                        "block_type": "text",
                    },
                ],
            ),
        ]
        
        response = await self.llm.achat(
            messages,
            max_tokens=self.max_contextual_tokens,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        return response.message.content

    def _parse_nodes(
    self,
    nodes: Sequence[BaseNode],
    show_progress: bool = False,
    **kwargs: Any,) -> List[BaseNode]:
        """Run the async implementation synchronously"""
        import asyncio
        return asyncio.run(self._aparse_nodes(nodes, show_progress, **kwargs))

    async def _aparse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse nodes to add context metadata."""
        # Group nodes by parent document for efficient processing#
        # TODO: investigate this code! this code assumes that 'nodes' is a list of all nodes and their parents in the same list. 
        # it also assumes it will be ordered with parents first then children
        # serious doubts about both assumptions - VERIFY
        # Check: how do other node parsers get the parent documents?
        nodes_by_parent: Dict[str, List[BaseNode]] = {}
        for node in nodes:
            if not node.source_node:
                continue
            parent_id = node.source_node.node_id
            if parent_id not in nodes_by_parent:
                nodes_by_parent[parent_id] = []
            nodes_by_parent[parent_id].append(node)

        output_nodes = []
        # TODO: this variable is unused. why? is that a problem? investigate.
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Adding context")

        node_context_jobs = []
        # Process each group of nodes sharing a parent document
        for parent_id, parent_nodes in nodes_by_parent.items():
            parent_content = parent_nodes[0].source_node.get_content()
            
            # Handle oversized documents
            if self.max_context_length:
                token_count = self._count_tokens(parent_content)
                if token_count > self.max_context_length:
                    message = f"Document {parent_id} is too large ({token_count} tokens) to be processed."
                    
                    if self.oversized_document_strategy == "truncate_first":
                        parent_content = self._truncate_text(parent_content, self.max_context_length, how='first')
                    elif self.oversized_document_strategy == "truncate_last":
                        parent_content = self._truncate_text(parent_content, self.max_context_length, how='last')
                    elif self.oversized_document_strategy == "warn":
                        logging.warning(message)
                    elif self.oversized_document_strategy == "error":
                        raise ValueError(message)
                    elif self.oversized_document_strategy == "ignore":
                        continue
                    else:
                        raise ValueError(f"Unknown oversized document strategy: {self.oversized_document_strategy}")

            # Generate context for each node
            for node in parent_nodes:
                for prompt, key in zip(self.prompts, self.keys):
                    node_context_jobs.append(
                        (node, self._agenerate_node_context(node, parent_content, prompt), key)
                    )

        # Run context generation jobs in parallel
        if node_context_jobs:
            contexts = await run_jobs(
                [job[1] for job in node_context_jobs],
                show_progress=self.show_progress,
                workers=self.num_workers,
            )

            # Apply generated contexts to nodes
            for (node, _, key), context in zip(node_context_jobs, contexts):
                node.metadata[key] = context
                output_nodes.append(node)

        return output_nodes

    @classmethod
    def class_name(cls) -> str:
        return "DocumentContextNodeParser"