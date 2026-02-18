import os
import streamlit as st
import PyPDF2
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(page_title="Chatbot")
st.title("RAG Chatbot")

DATA_FOLDER = "data"

def load_pdfs(folder_path):
    all_text = ""

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"

    return all_text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def build_vector_store():
    text = load_pdfs(DATA_FOLDER)
    chunks = chunk_text(text)

    embedding_model = load_embedding_model()
    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks, embedding_model

def retrieve(query, embedding_model, index, chunks, k=7):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

@st.cache_resource
def load_llm():
    return InferenceClient(api_key=HF_TOKEN)

index, chunks, embedding_model = build_vector_store()
client = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    #Retrieve relevant chunks
    retrieved_chunks = retrieve(user_input, embedding_model, index, chunks)
    context = "\n\n".join(retrieved_chunks)

    #Build conversation history string
    history_text = ""
    for msg in st.session_state.messages:
        history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    #Final Prompt
    prompt = f"""
You are a helpful AI assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Conversation History:
{history_text}

Answer:
"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )

            response = completion.choices[0].message.content
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
