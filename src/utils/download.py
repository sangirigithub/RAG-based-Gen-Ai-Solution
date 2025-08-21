### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

### import necessary libraries
from datasets import load_dataset
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

### 1. Data Download
def data_download():
    
    try:
        print('\n\tmod_Qns_download invoked .... ')
        
        # Load the Natural Questions dataset, selecting a portion for demonstration
        qna_raw = load_dataset('squad_v2')
        print('\tNum. of Docs Read:', len(qna_raw))
        print('TEST: ', qna_raw)
        
        df_qna_1 = pd.DataFrame(qna_raw['train'])
        df_qna_1.to_csv('train_SQuAD.csv', index=False)
        df_qna_2 = pd.DataFrame(qna_raw['train'])
        df_qna_2.to_csv('validation_SQuAD.csv', index=False)
        print('Stanford Question Answering Dataset Files Created!')

        return qna_raw
    except Exception as e:
        print(f"❌ Data Download or Saving Failed: {str(e)}")
        return None    

data_download()
### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

