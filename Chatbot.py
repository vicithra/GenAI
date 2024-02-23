import streamlit as st
from PyPDF2 import PdfReader, PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
OPENAI_API_KEY ="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"
# upload pdf files
st.header("My first lovable copilot")
with st.sidebar:
    st.title("Your pdf")
    file = st.file_uploader("Upload your pdf documents and start exploring by your questions", type="pdf ")
# extract text from pdf
if file is not None:
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
        # st.write(text)
#Break it into chunks
    text_splitter =RecursiveCharacterTextSplitter(
    separators="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
    )
    chunks=text_splitter.split_text(text)
    # st.write(chunks)
#generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#generate vectorstore
    vectorstores = FAISS.from_texts(chunks, embeddings)
#get user questions
    user_question = st.text_input("Enter your question")
#Do similarity search in vector store
    if user_question:
     match = vectorstores.similarity_search(user_question)
     # st.write(match)
#Define LLM
     llm=ChatOpenAI(
         openai_api_key=OPENAI_API_KEY,
         temperature=0,
         max_tokens=100,
         model_name="gpt-3.5-turbo"
     )
#output
     chain=load_qa_chain(llm,chain_type="stuff")
     response=chain.run(input_documents=match,question=user_question)
     st.write(response)


