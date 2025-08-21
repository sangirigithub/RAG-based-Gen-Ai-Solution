### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
"""
    Craft Prompt to Generate LLM Response for user Queries/Questions.
"""
### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

### import necessary libraries
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------

def craft_prompt():
    '''
    Craft a chat prompt template for LLM to Answer user Questions based on retrieved 'context'.

    Returns:
        ChatPromptTemplate: Composed prompt with system and human messages.
    '''
    try:
        # Prompt Template
        txt_system_prompt = ''' You are an Expert Assistant in Question-Answering Task.
            Please Answer User Questions in maximum 3 sentences, ONLY based on the below provided #CONTEXT.
            Please provide the most accurate Response based for each User Question.

            ONLY if you DO NOT find a Relevant Answer to the User Question from the #CONTEXT, 
            you Respond as "I do Not have an Answer to your Question at this point in time. 
            Is there Anything Else I can Help you with today!". 

            While generating answers to user questions, if you see a question which is inappropriate, unethical
            or anything against Responsible AI, or even outside of the context, Respond as per Responsible AI guidelines Only.
            And Always ensure to give Safe and Respectful Responses, free from vulgarity, bias or offensive content. 
            Responses should be Accurate, Fair and Considerate of all users, promoting a Trustworthy and Inclusive AI experience."

            # CONTEXT:
            {context}        
            '''            
        # Each element must be a 'message prompt template' or 'string' and not 'ChatPromptTemplate' object 
        # to be passed inside 'from_messages'. 'system_prompt_text' is a string with placeholder for 'context'
        system_prompt = SystemMessagePromptTemplate.from_template(txt_system_prompt)
        human_message = HumanMessagePromptTemplate.from_template('{input}')

        # Craft Prompt 
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])
        return prompt

    except Exception as e:
        print(f"❌ Prompt crafting error: {str(e)}")
        return "Default fallback prompt if needed"
    

### ----------------------------------------------------------------------------------------------
### ----------------------------------------------------------------------------------------------
