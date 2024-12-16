# Document Context Retrieval

## Summary
This repository contains the code from the anthropic blog post "contextual retrieval" (https://www.anthropic.com/news/contextual-retrieval)

it implements a custom llama_index Extractor class, which requires you to initialize it using a Document Store and an LLM to provide the context.

The recommended model is gpt-4o-mini, whose low cost and automatic prompt caching make it super cost efficient, and its plenty intelligent enough to handle the task.

In practice, I have not been able to get prompt caching to work with llama_index for some reason, and can only trigger prompt caching via the Anthropic python library.

Gemini flash 2.0 or any other fast cheap model would work as well.
Keep in mind input costs add up really fast with large documents.
you're going to pay for (doc_size * num_chunks) tokens for each document in input costs, and then (num_chunks * 100) or so for output tokens.

Make sure to keep document size below the context window of your model. pre-split the documents yourself if necessary.

## Usage

```python
docstore = SimpleDocumentStore()

llm = OpenRouter(model="openai/gpt-4o-mini")

# initialize the extractor
extractor = DocumentContextExtractor(document_store, llm)

storage_context = StorageContext.from_defaults(vector_store=self.vector_store,
                                                            docstore=docstore,
                                                            index_store=index_store)
index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            storage_context=storage_context,   
            transformations=[text_splitter, self.document_context_extractor]
        )

reader = SimpleDirectoryReader(directory)
documents = reader.load_data()

for doc in documents:
    self.index.insert(doc)
```

### custom keys and prompts

by default, the extractor adds a key called "context" to the document, and has a reasonable default prompt taken from the blog post, 
but you can pass in a list of keys and prompts like so:

```python
extractor = DocumentContextExtractor(document_store, llm, keys=["context", "title"], prompts=["Give the document context", "Provide a chunk title"])
```