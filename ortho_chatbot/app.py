import os
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
import streamlit as st


# Retrieve the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API Key is not set in the environment variables!")
os.environ['OPENAI_API_KEY'] = api_key

def document_data(query, chat_history):

    # Initialize the embeddings model
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

    # Define the persist directory (same as before)
    persist_directory = '/Users/igorhut/Documents/GitHub/ortho_chatbot/vector_dbs/tachdijans_db_20241203'
    if not os.path.exists(persist_directory):
        st.error(f"Persist directory does not exist: {persist_directory}")
        st.stop()

    # Reload the vector store from disk
    try:
        vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
        )
        st.success("Chroma vector store loaded successfully!")
    except ImportError as e:
        st.error(f"Chroma initialization failed due to missing dependencies: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error during Chroma vector store initialization: {e}")
        st.stop()
   # ConversationalRetrievalChain 
    llm_name = "gpt-4o"
    llm = ChatOpenAI(model_name=llm_name, temperature=0.0)
    qa = ConversationalRetrievalChain.from_llm(
       llm=llm, 
       retriever= vectorstore.as_retriever()
    )
    
    return qa({"question":query, "chat_history":chat_history})

if __name__ == '__main__':
    st.header("QA ChatBot")
    # ChatInput
    prompt = st.chat_input("Enter your questions here:")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if prompt:
       with st.spinner("Generating......"):
           output=document_data(query=prompt, chat_history = st.session_state["chat_history"])

           # Storing the questions, answers and chat history
           if "answer" in output:
                st.session_state["chat_answers_history"].append(output["answer"])
                st.session_state["user_prompt_history"].append(prompt)
                st.session_state["chat_history"].append((prompt, output["answer"]))
           else:
                st.error("No answer found in the response.")

    # Displaying the chat history
    if st.session_state["chat_history"]:
        for user_msg, assistant_msg in st.session_state["chat_history"]:
            st.chat_message("user").write(user_msg)
            st.chat_message("assistant").write(assistant_msg)
    