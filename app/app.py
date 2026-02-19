import os

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq


# ======================================================
# Configuration
# ======================================================

# Path to Chroma vector store (relative to project root: capstone_rag/chroma_db)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

CHROMA_COLLECTION_NAME = "research_papers"  # this is the collection name you already created

# Replace this with your actual Groq application programming interface [API] key
GROQ_API_KEY = "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"  # <-- put your key here, e.g. "gsk_...."


# ======================================================
# Initialize models and clients (load once)
# ======================================================

# Embedding model from sentence-transformers
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma persistent client and collection
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)

# Groq large language model [LLM] client
groq_client = Groq(api_key=GROQ_API_KEY)


# ======================================================
# Helper functions
# ======================================================

def retrieve_context(question: str, k: int = 3):
    """
    Retrieve top k relevant chunks from Chroma for a question.
    Returns:
        documents: list of text chunks
        ids: list of chunk identifiers
    """
    question_embedding = embed_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=k,
    )

    documents = results["documents"][0] if results.get("documents") else []
    ids = results["ids"][0] if results.get("ids") else []

    return documents, ids


def generate_answer(question: str, context_chunks):
    """
    Generate an answer from Groq large language model [LLM] using retrieved context.
    """
    context_text = "\n\n".join(context_chunks)

    prompt = (
        "You are an assistant that answers questions using only the given context.\n\n"
        "Context:\n"
        f"{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and concisely. "
        "If the answer is not in the context, say that you do not know."
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.2,
    )

    # Groq Python software development kit [SDK] returns a message object
    return response.choices[0].message.content


def rag_answer(question: str):
    """
    End to end retrieval augmented generation [RAG]:
        1. Retrieve top context chunks from vector database
        2. Call large language model [LLM] with question + context
        3. Return answer and source identifiers
    """
    context_chunks, source_ids = retrieve_context(question)

    if not context_chunks:
        return "I could not find any relevant context for this question.", []

    answer = generate_answer(question, context_chunks)
    return answer, source_ids


# ======================================================
# Streamlit user interface
# ======================================================

st.set_page_config(page_title="Research Paper Answer Bot")

st.title("Research Paper Answer Bot")
st.write(
    "Ask a question based on the indexed research papers in "
    "generative artificial intelligence [AI] and large language models [LLMs]."
)

user_question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not user_question.strip():
        st.write("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer, sources = rag_answer(user_question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources (chunk identifiers)")
        if sources:
            for src in sources:
                st.write(src)
        else:
            st.write("No sources were found for this answer.")

# Optional sidebar
st.sidebar.title("Project Info")
st.sidebar.write(
    "This is a retrieval augmented generation [RAG] demo built on top of a vector "
    "database of research papers in generative artificial intelligence [AI]."
)
