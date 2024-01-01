# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import requests
from streamlit_lottie import st_lottie
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import re


tab1, tab2 = st.tabs(["PDF", "Conversation with multi-PDF"])


def load_lottieUrl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_link = "https://lottie.host/931cf744-5c4d-418e-838f-9c5a28443482/YzPOvVtDnS.json"


def tab_pdf():
    load_dotenv()

    with st.container():
        left_column, right_column = st.columns((2, 1))
        with left_column:
            st.subheader("Welcome to ChatGPT PDF :alien:")
            st.title("PDF Extracted to ChatGPT")

        with right_column:
            st_lottie(lottie_link, height=300, key="coding")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        urls = set(re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text))
        for url in urls:
            st.write(url)

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


def get_pdf_text(pdf_docs):
    text = ""
    title = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            first_page = pdf_reader.pages[0]
            page_text = first_page.extract_text()
            text += page.extract_text()
            title_candidates = [
                line for line in page_text.splitlines() if line.isupper()
            ]
            title = max(title_candidates, key=len)

        st.write(title)
        urls = set(re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text))
        for url in urls:
            st.write(url)
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def tab_multipdf():
    load_dotenv()
    # st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.subheader("Your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
    )
    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    # with st.sidebar:
    #     st.subheader("Your documents")
    #     pdf_docs = st.file_uploader(
    #         "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
    #     )
    #     if st.button("Process"):
    #         with st.spinner("Processing"):
    #             # get pdf text
    #             raw_text = get_pdf_text(pdf_docs)

    #             # get the text chunks
    #             text_chunks = get_text_chunks(raw_text)

    #             # create vector store
    #             vectorstore = get_vectorstore(text_chunks)

    #             # create conversation chain
    #             st.session_state.conversation = get_conversation_chain(vectorstore)


def app():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    with tab1:
        tab_pdf()

    with tab2:
        tab_multipdf()


if __name__ == "__main__":
    app()
