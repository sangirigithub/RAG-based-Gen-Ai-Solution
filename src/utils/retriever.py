### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
'''
Use the Vector Store's 'retriever' interface to perform 'Similarity Search' with the user query,
Extract and return the most relevant document's content (or combine multiple documents if needed),
And pass that retrieved context to your language model for answer generation. '''

### import necessary libraries

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

def retrieve_context(query, vector_store, top_k: int=3):

    '''
    Retrieve relevant context documents for the question from the Vector DB.        
    Returns concatenated page contents as context.

    Args:
        query (str): User input question.
        vector_store: Loaded vector store object for similarity search.
        top_k (int): Number of top documents to retrieve.

    Returns:
        tuple: (retriever_object, concatenated_context_text) or (None, []) if no results.

    '''   

    if not query or not query.strip():
        return None, []  # no query, no retriever or context    

    try:
        # Using 'vector_store' as an interface between Vector DB (Chroma) and the RAG pipeline for retrieval 
        retriever = vector_store.as_retriever(search_kwargs={'k': top_k})  
        
        # Performs Similarity Searches and fetches Relevant Document Chunks for the user Query.
        # docs = retriever.get_relevant_documents(query)
        # qna_context = '\n\n'.join(doc.page_content for doc in docs)
        # 'BaseRetriever.get_relevant_documents' deprecated         
        retrieved_docs = retriever.invoke(query)        
        query_context = '\n\n'.join(doc.page_content for doc in retrieved_docs)

        if not retrieved_docs or not query_context:
            return None, []
        return retriever, query_context
    except Exception as e:
        print(f"❌ Retrieval failed: {str(e)}")
        return None, []
    
    
        
    
    
    

