# Document Context Retrieval

This repository contains the code from the anthropic blog post "contextual retrieval" (https://www.anthropic.com/news/contextual-retrieval)

it implements a custom llama_index Extractor class, which requires you to initialize it using a Document Store and an LLM to provide the context.

usage:

docstore=SimpleDocumentStore()

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