import os
import torch
import streamlit as st
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from qdrant_client import QdrantClient

from document_processors import load_multimodal_data
from memory_agent import MemoryAgent

# -----------------------
# Session state
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "index" not in st.session_state:
    st.session_state.index = None

# -----------------------
# Models
# -----------------------
@st.cache_resource
def initialize_llm():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    return HuggingFaceLLM(
        model_name=model_id,
        tokenizer_name=model_id,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16},
        context_window=2048,
        max_new_tokens=200,
    )

@st.cache_resource
def initialize_qdrant():
    return QdrantClient(path="./qdrant_data")

@st.cache_resource
def initialize_settings():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.llm = initialize_llm()
    Settings.text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)

# -----------------------
# Index creation
# -----------------------
def create_index(documents, client):
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="documents_collection"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# -----------------------
# App
# -----------------------
def main():
    st.set_page_config(layout="wide")
    initialize_settings()

    qdrant_client = initialize_qdrant()
    memory_agent = MemoryAgent(qdrant_client)

    st.title("Multimodal RAG with Long-Term Memory (Qdrant)")

    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

    if uploaded_files and st.button("Process Documents"):
        docs = load_multimodal_data(uploaded_files, Settings.llm)
        st.session_state.index = create_index(docs, qdrant_client)
        st.success("Documents indexed")

    # -----------------------
    # Chat + Query Handling
    # -----------------------
    if st.session_state.index:

        query_engine = st.session_state.index.as_query_engine(similarity_top_k=3)

        user_query = st.chat_input("Ask something")

        if user_query:
            print("Stage 1: User query received", flush=True)
            st.session_state.history.append(("user", user_query))

            print("Stage 2: Retrieving long-term memory from Qdrant", flush=True)
            memories = memory_agent.retrieve(user_query)
            print(f"Stage 3: Retrieved {len(memories)} memory items", flush=True)

            memory_text = "\n".join([m.payload.get("text", "") for m in memories])

            print("Stage 4: Building augmented query", flush=True)
            augmented_query = f"""
Known memory:
{memory_text}

User question:
{user_query}
"""

            print("Stage 5: Running vector retrieval and LLM generation", flush=True)
            response = query_engine.query(augmented_query)
            print("Stage 6: LLM response received", flush=True)

            answer = response.response

            print("Stage 7: Storing answer in long-term memory", flush=True)
            memory_agent.store(answer, metadata={"source": "assistant"})
            print("Stage 8: Memory storage completed", flush=True)

            st.session_state.history.append(("assistant", answer))

        # Display chat history
        for role, msg in st.session_state.history:
            st.chat_message(role).markdown(msg)

        if st.button("Clear Session Memory"):
            st.session_state.history = []

        if st.button("Delete Long-Term Memory"):
            memory_agent.delete_all()
            st.warning("Long-term memory cleared")

if __name__ == "__main__":
    main()
