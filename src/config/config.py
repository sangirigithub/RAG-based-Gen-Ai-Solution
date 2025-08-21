qna_trainData_filepath = r'../inputData/train_SQuAD.csv'
qna_valData_filepath = r'../inputData/validation_SQuAD.csv'
qna_responses_filepath = r'../outputData/llmResponses_Q&A.csv'

chroma_persist_dir = r'../chroma_db'

gen_embeddings_Flag = False 
# True - Will generate Text Embeddings for the records filtered as per below titles
# False - Will use the saved Text Embeddings from the Vector Store DB
titles = ['Computer', 'Idealism', 'Dog']



