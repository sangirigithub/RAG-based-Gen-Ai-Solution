### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
''' 
Embeddings: Ollama/HuggingFace
sentence-transformers/all-MiniLM-L6-v2 - great for fast, lightweight embeddings. 

Vector Store: Chroma/FAISS/Pinecone
Embed and index once.
Persist the index to disk.
Load the saved index on every app start or query session. '''

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

### import necessary libraries
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

def generate_VecEmbeddings(qna_chunks, txt_embed_model, persist_dir):
      
    """
    Generate vector embeddings and create a persistent vector store.

    Args:
        qna_chunks (list): List of document chunks with text content.
        txt_embed_model: Embedding model object to generate embeddings.
        persist_dir (str): Directory path to save the vector store persistently.

    Returns:
        None, Saves Chroma: Chroma vectorstore.
    """
    if not qna_chunks or len(qna_chunks) == 0:
        return # gracefully return nothing, don't Invoke Embeddings Model    
    
    try:
        # Assume qna_chunks is the list of QnA Chunks with 'page_content' having 'context' text
        chunks = [chunk.page_content for chunk in qna_chunks]
    
        # Create local Chroma 'vector_store' from texts and embeddings and save in ./chroma_db directory, 
        # Ready for efficient semantic search and retrieval.
        # Example: call embedding API
        vector_store = Chroma.from_texts(texts=chunks, embedding=txt_embed_model, persist_directory=persist_dir)
    except Exception as e:
        print(f"❌ Failed to Store Embeddings: {str(e)}")

    # Persist the index to disk locally
    # vector_store.persist()  # Chroma 0.4.x - docs are now automatically persisted.

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

