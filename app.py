from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import load_chain
from langchain.llms import OpenAI
import streamlit as st
import pyautogui
import os, shutil

def delete_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' successfully deleted.")
    except Exception as e:
        print(f"Error deleting directory '{directory_path}': {e}")

st.set_page_config(page_title="Query any Pdf", page_icon="📄")

st.title("📄 PDF Query Bot 📄")
st.write("Made with ❤️ by Mainak")

def return_response(query,document_search,chain):
    query = query
    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    return result

uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])

# API key input box
api_key = st.text_input("Enter Your OpenAI API Key",type="password")

if not  uploaded_file:
    try:
        delete_directory('faiss_index')
    except:
        pass

if st.button('Submit'):
    if api_key:
        if uploaded_file is not None:
            # Read text from the uploaded file
            os.environ["OPENAI_API_KEY"] = api_key
            with st.spinner('Wait for it...'):
                pdfreader = PdfReader(uploaded_file)
                # read text from pdf
                raw_text = ''
                for i, page in enumerate(pdfreader.pages):
                    content = page.extract_text()
                    if content:
                        raw_text += content

                text_splitter = CharacterTextSplitter(
                    separator = "\n",
                    chunk_size = 800,
                    chunk_overlap  = 200,
                    length_function = len,
                )
                texts = text_splitter.split_text(raw_text)
                embeddings = OpenAIEmbeddings()
                document_search = FAISS.from_texts(texts, embeddings)
                document_search.save_local("faiss_index")
        else:
            st.warning("Please enter your Pdf File")
    else:
        st.warning("Please enter your API key")
if os.path.exists("faiss_index"):
    # if st.checkbox("chat"):
        if api_key:
            if uploaded_file is not None:
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display chat messages from history on app rerun
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                if prompt := st.chat_input("What is up?          Type 'exit' for leave the chat"):
                    # Display user message in chat message container
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                os.environ["OPENAI_API_KEY"] = api_key
                embeddings = OpenAIEmbeddings()
                document_search = FAISS.load_local("faiss_index", embeddings)
                chain = load_qa_chain(OpenAI(), chain_type="stuff")
                if prompt is None:
                    re='Ask me anything about the pdf'
                elif prompt=='exit':
                    delete_directory('faiss_index')
                    pyautogui.hotkey('f5') #Simulates F5 key press = page refresh
                else:
                    with st.spinner('Typping...'):
                        re=return_response(str(prompt),document_search,chain)
                response = f"PDF Mate: {re}"
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning("Please enter your Pdf File")
        else:
            st.warning("Please enter your API key")
else:
    pass