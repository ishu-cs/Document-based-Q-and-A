from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
#from langchain_community.document_loaders import PyPDFLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
import pandas as pd
from langchain.vectorstores import FAISS
import streamlit as st
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
folder_path=os.getcwd()

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",model_kwargs={"device":"cpu"})
llm=CTransformers(
        model='D:\Data-Science\Gen_AI\LLM\Langchain\email-generator\models\llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={'max_new_tokens':256,
                'temperature':0}
    )

def document_loader():
    loader=DirectoryLoader(f'{folder_path}\\Data',glob="./*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_documents():
    documents=document_loader()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    texts=text_splitter.split_documents(documents)
    return texts

if os.path.exists(f"{folder_path}\\faiss_index_docQnA"):
    print("Vector Store exists")
    VectorStore=FAISS.load_local(f"{folder_path}\\faiss_index_docQnA",embeddings,allow_dangerous_deserialization=True)
else:
    print("Vector Store does not exists. Creating first time hence will take few minutes")
    os.makedirs(f"{folder_path}\\faiss_index_docQnA")
    print(f"The Vector store creation is in progress ....")
    texts=split_documents()
    VectorStore=FAISS.from_documents(texts,embeddings)
    print(f"The Vector store creation is done ....")
    VectorStore.save_local(f"{folder_path}\\faiss_index_docQnA")
    print(f"Vector DB is saving to local drive at {folder_path}\\faiss_index_docQnA Finished!!")



questions =[]
answers=[]
ref_doc=[]
page=[]

def generate_qa_chain(VectorStore):
    retriver=VectorStore.as_retriever(search_kwargs={"k":3})
    qa_chain=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriver,input_key="question")
    return qa_chain,retriver


def get_bot_response():
    query=form_input
    questions.append(query)
    qa_chain,retriever = generate_qa_chain(VectorStore)
    ans=qa_chain(query)
    response=ans.get('result')
    answers.append(response)
    ref_doc.append(retriever.get_relevant_documents(query)[0].metadata.get('source'))
    page.append(retriever.get_relevant_documents(query)[0].metadata.get('page')+1)
    return response

st.set_page_config(
    page_title="Document based Question and answers",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.header("GENERATE EMAILS")

form_input = st.text_area('Question',height=200)

submit=st.button("Generate")


## When the 'Generate' button is clicked run below code
if submit:
    #st.write("Response")
    st.write(get_bot_response())