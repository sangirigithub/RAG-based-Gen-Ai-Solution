### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
'''
CSV format file created from the downloaded datset - load_dataset('squad_v2'); 
Train and Validation datasets available.

Records in the dataset have 'page_content like single string for each record, although columns are present,
they are not in the format of list or dictionary, but just a string, so parsing is needed to extract 
'context' field as the main content which is needed to implement the RAG solution for Question Answering Model.
Extracting 'title' as metadata just to see what kind of Docs are present in the dataset. '''

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

### import necessary libraries
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import random

from config import config as cg

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

### 1. Data Loading
def data_loader(filePath):
    '''   
    Use CSV Data Loader to load the CSV file extracted from the download and 'parse' the documents/records.

    Args:
        filePath (str): Folder Path of the 'train' input CSV File.
        lst_titles (list): List of titles to filter relevant documents.
        
    Returns:
        list: List of chunked Document objects ready for embedding.
    '''    
    try:
        cvs_loader = CSVLoader(file_path=filePath, encoding='utf-8')  
        qna_raw = cvs_loader.load() 
        if not qna_raw:
            print("⚠️ No documents found during ingestion.")        
        return qna_raw

    except FileNotFoundError:
        print(f"❌ File not found: {filePath}")
        return []
    except Exception as e:
        print(f"❌ Loading error: {str(e)}")
        return []   
    
### ------------------------------------------------------

### 2. Data Parsing 
    ''' For Each Document/Record, Extract 'context' field for QnA and 'title' for analysis purpose. '''
def data_parser(doc_parse):

    try:
        text = doc_parse.page_content

        ## Extract 'title' using regex
        title_match = re.search(r"title:\s*(.*)", text)
        title = title_match.group(1).strip() if title_match else "Unknown"
        
        ## Extract 'context' - text content between 'context:' and 'question:' 
        context_match = re.search(r"context:\s*(.*?)\nquestion:", text, re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""

        ## Overwrite doc.page_content with the 'context' to be used for retrieval
        doc_parse.page_content = context

        ## Add extracted title as metadata
        doc_parse.metadata["title"] = title

        return doc_parse    
    except Exception as e:
        print(f"❌ Parsing error: {str(e)}")
        return []  
### ------------------------------------------------------

### 3. Data Chunking
def text_chunker(qna_parsed):

    try:
        ## Dataset Max text length: 3899 Chars 
        ##  Average document length: 942.75 Chars
        ## 15% of chunk Size as chunk Overlap 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=90) 

        # list of strings (texts), Extracted 'page_content'
        qna_texts = [doc.page_content for doc in qna_parsed] 

        # Input - List of strings, Output - List of Document Objects
        # Use split_documents if Input is List of Document Objects, Output will be smaller Document Objects
        qna_chunks = text_splitter.create_documents(qna_texts)

        return qna_chunks

    except Exception as e:
        print(f"❌ Chunking error: {str(e)}")
        return []  
### ------------------------------------------------------

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

def data_ingestion(filePath, titles):
    
    try:
        qna_raw = data_loader(filePath)
        print('\tNum. of Docs Read:', len(qna_raw)) # 130319
        # print('Sample "RAW" QnA Docs:\n ', qna_raw[12000]) #[704:706])
        print('------------------------------------------')         
        
        qna_parsed = [data_parser(doc) for doc in qna_raw]
        print('\tNum. of Docs Parsed:', len(qna_parsed)) # 130319
        # print('Sample "PARSED" QnA Docs:\n ', qna_parsed[12000]) #[704:706])
        print('------------------------------------------') 

        # Shuffle docs list for diversity
        random.shuffle(qna_parsed)

        qna_filtered = [doc for doc in qna_parsed if doc.metadata.get('title') in titles]
        # qna_filtered = [doc for doc in qna_parsed if doc.metadata.get('title')] # use this when 'gen_embeddings_Flag = True'
        print('\tNum. of Docs Filtered:', len(qna_filtered)) # 130319/ 
        print('------------------------------------------') 
        
        qna_chunks = text_chunker(qna_filtered)
        print('\tNum. of Chunked Docs:', len(qna_chunks))
        # print('Sample "CHUNKED" QnA Docs:\n ', qna_chunks[5:8]) # 234686/
        print('--------------------------------------------')

        return qna_chunks
    except Exception as e:
        print(f"❌ Ingestion error: {str(e)}")
        return []  
        
### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
