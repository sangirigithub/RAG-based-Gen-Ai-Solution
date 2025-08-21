### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
"""
    Generate LLM Response as an Answer for user Queries/Questions.
"""
### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

### import necessary libraries
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

def generate_answer(query, retriever, query_context, llm, API_key, prompt):

    """
    Generate answer from LLM using retrieved context and user query.

    Args:
        query (str): User question string.
        retriever: Retriever object used for document retrieval.
        query_context (str): Retrieved context string.
        llm: Large Language Model instance.
        API_key (str): API key for accessing the LLM.
        prompt: Prompt template to structure LLM input.

    Returns:
        str: Generated answer (fallback message if no context).
    """

    # Generate an answer for the question using the retrieved context and Large Language model.     
    # If no context retrieved, return fallback message
    if not query_context:
        return "Sorry, I couldn't find relevant information to answer your question."
    
    try:
        # Chain to Create the 'context' based on the user Query passed to 'prompt' and passes it to the Model
        doc_chain = create_stuff_documents_chain(llm, prompt)

        # RAG Chain to use 'retriever' to fetch Relevant Documents from Vectore Store DB based on Similar Search
        rag_chain = create_retrieval_chain(retriever, doc_chain)

        # Invoke the RAG Chain passing the user 'query' to Generate LLM Response as an Answer to user Question
        llm_response = rag_chain.invoke({'context': query_context, 'input': query})
        # Optional debug prints
        # print("LLM raw response:", llm_response)

        answer = llm_response.get('answer', "Sorry, no answer was generated.") # llm_response['answer']

    except Exception as e:
        print(f"❌ LLM Invocation failed: {e}")
        return "Error: Failed to generate an Answer. Please check LLM Model & API key and try again."    

    return answer 

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
