from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
import google.generativeai as genai
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():
    prompt_template = '''
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context just say, "answer is not available in the context", don't provide the worng answer.\n\n
    Context: \n {context} ? \n
    Question: \n {question} \n
    Answer:
    '''
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply :", response["output_text"])


def main():
    st.set_page_config(page_title="EaseReader", page_icon=":books:")
    st.header("Chat with multiple PDF's :books:")
    user_question = st.text_input("Ask a question about your document: ")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.subheader("Your Docs")
        pdf_docs = st.file_uploader(
            "Upload your PDF's Here", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                get_vectorstore(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
