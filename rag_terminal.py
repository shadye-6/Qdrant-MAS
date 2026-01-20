import os
import torch

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

from qdrant_client import QdrantClient

from document_processors import load_multimodal_data
from memory_agent import MemoryAgent


# ============================================================
# Model initialization
# ============================================================

def initialize_llm():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    return HuggingFaceLLM(
        model_name=model_id,
        tokenizer_name=model_id,
        device_map="cpu",
        model_kwargs={"torch_dtype": torch.float32},
        max_new_tokens=200,
    )


def initialize_settings():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Settings.llm = initialize_llm()

    Settings.text_splitter = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=50
    )


def initialize_qdrant():
    return QdrantClient(path="./qdrant_data")


# ============================================================
# Index creation
# ============================================================

def create_index(documents, client):
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="documents_collection"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)


# ============================================================
# Helpers
# ============================================================

def print_stage(msg):
    print(msg, flush=True)


def load_documents_from_folder(folder_path):
    fake_files = []

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            fake_files.append(open(full_path, "rb"))

    docs = load_multimodal_data(fake_files, Settings.llm)

    for f in fake_files:
        f.close()

    return docs


def truncate_text(text, max_chars=1200):
    if len(text) > max_chars:
        return text[:max_chars] + "\n[truncated]"
    return text


# ============================================================
# Main CLI Application
# ============================================================

def main():
    print("\nInitializing system...\n")

    initialize_settings()
    qdrant_client = initialize_qdrant()
    memory_agent = MemoryAgent(qdrant_client)

    folder_path = input("Enter path to documents folder: ").strip()

    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return

    print_stage("Stage A: Loading documents...")
    documents = load_documents_from_folder(folder_path)
    print_stage(f"Stage B: Loaded {len(documents)} document chunks")

    print_stage("Stage C: Creating vector index in Qdrant...")
    index = create_index(documents, qdrant_client)
    print_stage("Stage D: Index creation completed")

    query_engine = index.as_query_engine(similarity_top_k=3)

    print("\nSystem ready. Type your questions below.")
    print("Type 'exit' to quit.")
    print("Type 'clear_memory' to erase long-term memory.\n")

    while True:
        try:
            user_query = input("\nUser: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not user_query:
            continue

        if user_query.lower() == "exit":
            print("Exiting.")
            break

        if user_query.lower() == "clear_memory":
            memory_agent.delete_all()
            print("Long-term memory cleared.")
            continue

        print_stage("Stage 1: User query received")

        print_stage("Stage 2: Retrieving long-term memory from Qdrant")
        memories = memory_agent.retrieve_text(user_query)
        print_stage(f"Stage 3: Retrieved {len(memories)} memory items")

        memory_text = "\n".join(m.payload.get("text", "") for m in memories)
        memory_text = truncate_text(memory_text)

        print_stage("Stage 4: Building augmented query")

        augmented_query = f"""Known memory:
{memory_text}

User question:
{user_query}
"""

        print_stage("Stage 5: Running vector retrieval and LLM generation")

        try:
            response = query_engine.query(augmented_query)
            answer = response.response
        except Exception as e:
            print("LLM error:", e)
            continue

        print_stage("Stage 6: LLM response received")

        print_stage("Stage 7: Storing answer in long-term memory")
        memory_agent.store_text(answer, metadata={"source": "assistant"})
        print_stage("Stage 8: Memory storage completed")

        print("\nAssistant:\n")
        print(answer)


if __name__ == "__main__":
    main()
