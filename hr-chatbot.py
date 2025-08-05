# HR Chatbot with FAISS + GPT-4o via GitHub Token

from dotenv import load_dotenv
import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()  # Load environment variables from .env file


def build_chat_history(chat_history_list):
    """
    Converts a list of (question, answer) tuples into LangChain's chat history format.
    """
    chat_history = []
    for question, answer in chat_history_list:
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
    return chat_history


def query(question, chat_history_list):
    """
    Handles semantic retrieval + chat with context awareness.
    Uses FAISS vector DB and GitHub-token-based GPT-4o.
    """
    chat_history = build_chat_history(chat_history_list)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Load the FAISS vector database
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # GPT-4o from GitHub token
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.github.ai/inference"
    )

    # Prompt for rephrasing user's question in context of history
    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, db.as_retriever(), condense_question_prompt
    )

    # QA Prompt template
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an assistant for question-answering tasks on HR Policy. "
            "Use the following retrieved context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentences maximum. Keep it concise.\n\n{context}"
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Invoke the retrieval chain
    return retrieval_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })


def show_ui():
    """
    Streamlit-based UI for interacting with the HR chatbot.
    """
    st.set_page_config(page_title="HR ChatBot")
    st.title("HR Chatbot 🤖")
    st.image("c4x-cbt.jpg")
    st.subheader("Ask me anything about HR policies!")

    # Initialize chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input prompt
    if prompt := st.chat_input("Enter your HR Policy related Query:"):
        with st.spinner("Working on your query..."):
            response = query(question=prompt, chat_history_list=st.session_state.chat_history)

        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(response["answer"])

        # Update session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        st.session_state.chat_history.append((prompt, response["answer"]))


if __name__ == "__main__":
    show_ui()
