### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
''' 
SmartVaultHub: Knowledge Retrieved, Answers Delivered 
(RAG(Retrieval-Augmented Generation) - Based Question Answering Solution) 

Modules in this RAG-based Q&A Model:
•	Loading and parsing the dataset properly,
•	Extracting and cleaning useful fields like context and title,
•	Chunking the data for effective retrieval,
•	Creating embeddings with an open-source model,
•	Building a vector store for efficient search,
•	Integrating with a language model to generate answers based on retrieved context

'''
### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

### import necessary libraries
import os
import time
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from config import config as cg
from utils import ingest as ig 
from utils import embeddings as em
from utils import generator as gn
from utils import prompt as pm
from utils import retriever as rt

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

### Initialize Parameter Variables
env_path = r'config/.env'
load_dotenv(dotenv_path = env_path) 
# os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
apiKey = os.getenv("GROQ_API_KEY")

# Initialize the 'embedding' model, replace with any ohter choosen embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

### Initialize Parameter Variables / Populate Input data from the UI
titles = cg.titles #lst_titles=['Computer'] 
gen_embeddings_Flag = cg.gen_embeddings_Flag
# True - Will generate Text Embeddings for the records filtered as per below titles (took 4Hrs to Generate Embeddings)
# False - Will use the saved Text Embeddings from the Vector Store DB

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

def rag_pipeline(user_question, llm_model, apiKey, top_k_docs):

    """
    Full RAG query pipeline: Loads Data, Processes, Performs Chunking, Creates Text Embeddings and Vector Store DB,
    Loads Vector Store DB, Retrieves Query Context, Generate LLM Response as an Answer to user's Question.
    
    Args:
        user_question (str): User input question.
        llm_model (str): Selected LLM model name.
        apiKey (str): API key for LLM access.
        top_k_docs (int): Number of top documents to retrieve for context.

    Returns:
        str: The generated response as answer to user's input question.
    """

    if gen_embeddings_Flag:
        print('Generate Embeddings Flag is set to True, so Embeddings will be generated and saved to Chroma Vector Store DB!')
        ### 1. ingest - Load, Parse and Chunk Dataset
        print('\n\t Invoking "Module - ingest - data_loader, data_parser, text_chunker and data_ingestion" .... ') 
        qna_docs_chunks = ig.data_ingestion(cg.qna_trainData_filepath, titles)
        if not qna_docs_chunks:
            print("⚠️ No documents to embed after ingestion.")
            return {"error": "No documents found to generate Vector Embeddings."}
  
        ### 2. embeddings - Generate Embeddings & Create Vector DB, Load Vector DB & Retrieve Context
        print('\n\t Invoking "Module - embeddings - generate_VecEmbeddings" .... ')     

        if not qna_docs_chunks: # Not 'None'
            print("⚠️ No chunks produced by ingestion, aborting embedding generation.")
            return {"error": "No documents found to generate embeddings."}
        
        try:
            em.generate_VecEmbeddings(qna_docs_chunks, embedding_model, cg.chroma_persist_dir)
            # em.generate_VecEmbeddings(qna_docs_chunks[15:20], embedding_model, cg.chroma_persist_dir)
            print('Embeddings Generated and Chroma Vector DB created!')
        except Exception as e:
            print(f"❌ Embedding failed: {str(e)}")
            return {"error": "Failed to generate embeddings. Please retry."}

    else:
        print('Generate Embeddings Flag is set to False, so Model will use Embeddings saved in Chroma Vector Store DB!')        

    print('----------------------------------------------------------------------------------------')

    ### 3. retriever - retrieve the 'context'    
    print('\n\t Invoking "Module - retriever - load_vector_store, retriever and retrieve_context" .... ')
    
    start = time.process_time()   

    ##### Run RAG Pipeline Stages
    # Load persist 'vector_store' to use for retrieval by loading the same
    # embedding_model: Text Embedding model used while generating the Vector Embeddings
    vector_store = Chroma(persist_directory=cg.chroma_persist_dir, embedding_function=embedding_model)
    retriever, qna_context = rt.retrieve_context(user_question, vector_store, top_k_docs)

    if not retriever or not qna_context: # Not 'None'
        print("⚠️ Retriever or retrieved context is Empty.")
        return {"answer": "No documents found in the knowledge base relevant to your query."}, []
    print('----------------------------------------------------------------------------------------')

    ### 4. generator - Generate LLM Response as an 'answer' to the 'user_question/query
    print('\n\t Invoking "Module - generator - generate_answer" .... ')
    # Generate an answer for the question using the retrieved context and Large Language model.        
    llm = ChatGroq(groq_api_key=apiKey, model_name=llm_model) # Gemma2-9b-It/Mistral-7B/Mixtral 8×7B
    qna_prompt = pm.craft_prompt()    
    print("Prompt for LLM:\n", qna_prompt)
    print('----------------------------------------------------------------------------------------')

    try:
        answer = gn.generate_answer(user_question, retriever, qna_context, llm, apiKey, qna_prompt)    
        
        print('----------------------------------------------------------------------------------------')
        print('User Question:')
        print(user_question)
        print('Answer LLM Generated using RAG Pipeline:')
        print(answer)     
        print('Context:')
        print(qna_context)   

        print(f"Total Pipeline Latency: {round(time.process_time()-start, 2)}")
        print('----------------------------------------------------------------------------------------')
        
        return answer, qna_context
    except Exception as e:
        print(f"❌ LLM call failed: {str(e)}")
        return {"error": "Failed to generate an answer! Please check LLM Model & API key and Retry!"}

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

def main():
    '''    
    Perform Retrieval-Augmented Generation (RAG) to answer user's Questions.
    
    This function takes a user question, performs a similarity search on the vector store
    to retrieve relevant document chunks, then passes the retrieved context and question
    to a Large Language Model (LLM) to generate response as a final answer.    
    '''   

    # Streamlit Sidebar for settings
    st.markdown('<h1 style="font-size:30px;">RAG-based Question Answering Solution</h1>', unsafe_allow_html=True)
    
    # Sidebar Inputs
    ### 1. User Input - LLM Model
    llm_model = st.sidebar.selectbox(
        'Choose one of the LLM Models',
        ['Llama3-8b-8192', 'Gemma2-9b-It', 'gemma3:27b', 'mistral']
    )    
    # llm_model = 'Llama3-8b-8192'
    st.sidebar.markdown('''<h3 style="font-size:10px;">*Groq - Llama3-8b-8192, Gemma2-9b-It; 
                Ollama - gemma3:27b, mistral; HuggingFace - Ministral-8B-Instruct-2410</h3>''', unsafe_allow_html=True)
    
    ### 2. User Input - LLM API Key
    apiKey = st.sidebar.text_input('API Key for the LLM choosen', type='password')
    st.sidebar.markdown('''
    <h3 style="font-size:10px;">
    *Create API key and access available Open-source Models by visiting:<br><br>
    1. <a href="https://console.groq.com/keys" target="_blank">GroqCloud API Keys Console</a><br><br>
    2. <a href="https://ollama.com/library" target="_blank">Ollama Model Library</a><br><br>
    3. <a href="https://huggingface.co/settings/tokens" target="_blank">HuggingFace Models</a>
    </h3>''', unsafe_allow_html=True)  
    
    ### 3. User Input - top-k Docs
    top_k_docs = st.sidebar.slider( 
        'No. of top-k Docs to Retrieve Context:',
        min_value=1,
        max_value=4,
        value=2,   # Default Value
        step=1
    ) # top_k = 2 

    ### 4. User Input - Question
    user_question = st.text_input('What Question do you have today: ')
    # user_question = 'What translates Programming languages into machine code?'

    with st.form(key='qna_form'):       
        # st.write(f"user_question: '{user_question}'")
        # st.write(f"llm_model: '{llm_model}'")
        # st.write(f"apiKey: '{apiKey}'")
        
        # Compute boolean condition for all_filled by checking if all the inputs are 'non-empty'
        all_filled = bool(user_question.strip()) and bool(llm_model.strip()) and bool(apiKey.strip())

        ### 5. Submit Button
        # Submit Button Enables dynamically once all the inputs are filled
        bt_submit = st.form_submit_button(label='Ask LLM', disabled=not all_filled)    

    print('----------------------------------------------------------------------------------------')
        
    if bt_submit:
        if user_question and llm_model and apiKey:
            try:
                with st.spinner('Generating answer...'):
                    print('\n\n Invoking "Module - rag_pipeline" .... ')                    
                    answer, qna_context = rag_pipeline(user_question, llm_model, apiKey, top_k_docs)

                    st.markdown('<u>**Please find below LLMs Reponse to your Question:(Model uses saved Vector Store DB)**</u>', unsafe_allow_html=True)
                    st.write(answer)

                    with st.expander("Context passed to the LLM Model"):                        
                        st.write(qna_context)

            except Exception as e:
                st.error(f'Error during Model Execution: {e}')    
        else:
            st.warning('Please fill all input fields before submitting!')

if __name__ == "__main__":
    print('***********************************************************************************')
    print('\n\n Invoking Module - main() - RAG-based Question Answering Solution .... ')
    main()
    print('***********************************************************************************')

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------



