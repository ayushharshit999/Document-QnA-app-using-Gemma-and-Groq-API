## Firstly, I imported the necessary libraries.
# I used streamlit since it builds interactive web apps allowing data scientists to showcase their models without the need
# of webdev skills.
#I used faiss cpu for vector embedding, pypdf to read documents from the pdf.
#I also used python-dotenv to call the environment variables.

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()

## loading the GROQ And Google OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

## In groq we have chatgroq which i used to create a chatbot using the api key which was totally free
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

# I have used chat prompt template to create my own custom prompt template
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

## I have created a function name vector embedding below where i will read all my pdf files,then convert them into chunks, then
## apply embeddings and then i will store them in a vector store DB "FAISS".
def vector_embedding():

# I have kept this vector store db in different variables in my session state so that i will be able to use it anywhere it is
# required.
    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./selfhelpbooks") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings




# taking input from the user
prompt1=st.text_input("Enter Your Question From Doduments")

# I have created a button which is responsible in creating vector store
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

# I have also imported time to record the time it takes to give the output response
import time


# Here whenever the user will write any text and press enter. Document chain will be created with 2 parameters- llm model and prompt
# Then I retrieved the vector store db and document chain combined as retrieval chain.
# Then I called the invoke function to finally get the response which i have displayed in my streamlit app.
if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

# Whenever gemma model provides us the response, it also provides us some kind of context in return which I have displayed
# using streamlit expander below
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
