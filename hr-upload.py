import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

def upload_htmls():
    """
    1. Reads recursively through the folder 'hr-policies'.
    2. Loads HTML pages.
    3. Splits them into chunks.
    4. Converts chunks to vector embeddings.
    5. Saves the FAISS vector DB locally.
    """
    loader = DirectoryLoader(path="E:/hr-policies", glob="**/*.html", show_progress=True)
    documents = loader.load()
    print(f"{len(documents)} HTML pages loaded.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(split_documents)} chunks.")
    print("Example metadata:", split_documents[0].metadata)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_documents, embeddings)
    db.save_local("faiss_index")
    print("✅ FAISS vector store saved locally.")

def faiss_query():
    """
    Loads the FAISS DB, performs a semantic search, and uses GPT-4o (via GitHub token) to answer the query.
    """
    query = "Explain the Candidate Onboarding process."

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings)

    retriever = db.as_retriever()

    # Prompt Template
    prompt = PromptTemplate.from_template("""
        Based on the following HR documents, answer this question clearly and concisely:

        Question: {input}

        Documents:
        {context}

        Only return the final answer, no extra formatting or commentary.
    """)

    # LLM setup using GitHub token (GPT-4o)
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.github.ai/inference"
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": query})
    print("💬 GPT-4o Response:\n")
    print(response["answer"])

if __name__ == "__main__":
    # Run once to upload and index the documents
    upload_htmls()

    # Run this to query the vector DB and use GPT-4o to generate an answer
    #faiss_query()
