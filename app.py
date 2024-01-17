from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import load_chain
from langchain.llms import OpenAI
import streamlit as st
import openai
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import google.generativeai as genai

import os, shutil


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text,method):
    if method=='Google-Gemini':
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        chunks = text_splitter.split_text(text)
    else:
        text_splitter = CharacterTextSplitter(separator = "\n",chunk_size = 1000,chunk_overlap  = 300,length_function = len)
        chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks,method):
    try:
        if method=='Google-Gemini':
            embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        else:
            embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except:
        st.warning("Wrong API, give a valid API")


def get_conversational_chain(method):
        
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    if method=='Google-Gemini':
        model = ChatGoogleGenerativeAI(model="gemini-pro",
                            temperature=0.3)
    else:
        model= OpenAI()
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def user_input(user_question,method):
    if method=='Google-Gemini':
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    else:
        embeddings = OpenAIEmbeddings()
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(method)

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    return response




def delete_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' successfully deleted.")
    except Exception as e:
        print(f"Error deleting directory '{directory_path}': {e}")

def return_response(query,document_search,chain):
    query = query
    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    return result




st.set_page_config(page_title="Query any Pdf", page_icon="📄")

st.title("📄 PDF Query Bot 📄")
st.write("Made with ❤️ by Mainak")
with st.sidebar:
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit Button", accept_multiple_files=True,type=['pdf'])
    option = st.selectbox('Select a Model(choose OpenAI for best results)',('OpenAI', 'Google-Gemini'))
    if option=='OpenAI':
        api_key = st.text_input("Enter Your OpenAI API Key",type="password")
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        api_key = st.text_input("Enter Your Google-Gemini API Key",type="password")
        os.environ["google_API_KEY"] = api_key
        genai.configure(api_key=os.getenv("google_API_KEY"))
if not  pdf_docs:
    try:
        delete_directory('faiss_index')
    except:
        pass
with st.sidebar:
    if st.button('Submit'):
        if api_key:
            if pdf_docs is not None:
                # Read text from the uploaded file
                os.environ["OPENAI_API_KEY"] = api_key
                with st.spinner('Wait for it...'):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text,option)                   
                    get_vector_store(chunks,option)
            else:
                st.warning("Please enter your Pdf File")
        else:
            st.warning("Please enter your API key")

if os.path.exists("faiss_index"):
        if api_key:
            if pdf_docs is not None:
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display chat messages from history on app rerun
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                if prompt := st.chat_input("What is up?"):
                    # Display user message in chat message container
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": prompt})
                # os.environ["OPENAI_API_KEY"] = api_key
                if prompt is None:
                    re='Ask me anything about the pdf'
                else:
                    with st.spinner('Typping...'):
                        re = user_input(str(prompt),option)
                        re = re["output_text"]
                        # re=return_response(str(prompt),document_search,chain)
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

