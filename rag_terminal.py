import os
import torch

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llm_llama_cpp import LlamaCppLLM

from qdrant_client import QdrantClient

from document_processors import load_multimodal_data
from memory_agent import MemoryAgent
from agents.graph_store import GraphStore
from agents.orchestrator import OrchestratorAgent
from agents.graph_extractor import GraphExtractor


# ============================================================
# Model initialization
# ============================================================

def initialize_hf_llm():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    return HuggingFaceLLM(
        model_name=model_id,
        tokenizer_name=model_id,
        device_map="auto",
        max_new_tokens=256,
    )


def initialize_llm():
    return LlamaCppLLM(model_path="models/mistral.gguf")


def initialize_settings():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Settings.llm = initialize_hf_llm()

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


def load_documents_from_folder(folder_path, graph_extractor, graph_store):
    fake_files = []

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            fake_files.append(open(full_path, "rb"))

    docs = load_multimodal_data(fake_files, Settings.llm, graph_extractor, graph_store)

    for f in fake_files:
        f.close()

    return docs


def truncate_text(text, max_chars=1200):
    if len(text) > max_chars:
        return text[:max_chars] + "\n[truncated]"
    return text


def extract_keywords(query, max_words=5):
    words = [
        w.lower().strip(".,!?")
        for w in query.split()
        if len(w) > 3
    ]
    return words[:max_words]


# ============================================================
# Main CLI Application
# ============================================================

def main():
    print("\nInitializing system...\n")

    initialize_settings()

    vector_llm = Settings.llm
    graph_llm = initialize_llm()

    graph_extractor = GraphExtractor(graph_llm)

    qdrant_client = initialize_qdrant()
    memory_agent = MemoryAgent(qdrant_client)
    graph_store = GraphStore()
    router = OrchestratorAgent()

    folder_path = input("Enter path to documents folder: ").strip()

    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return

    print_stage("Stage A: Loading documents...")
    documents = load_documents_from_folder(folder_path, graph_extractor, graph_store)
    print_stage(f"Stage B: Loaded {len(documents)} document chunks")

    print_stage("Stage C: Creating vector index in Qdrant...")
    index = create_index(documents, qdrant_client)
    print_stage("Stage D: Index creation completed")

    # ✅ ENABLE QUERY ENGINE
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

        print_stage("Stage 4: Routing query")
        route = router.route(user_query)
        print_stage(f"Stage 4b: Route determined: {route}")

        print_stage("Stage 4c: Retrieving graph context")
        graph_context = ""
        if route == "graph":
            keywords = extract_keywords(user_query)
            graph_results = graph_store.query(keywords)
            graph_context = "\n".join(graph_results)
            print_stage(f"Stage 4d: Retrieved {len(graph_results)} graph relationships")
        else:
            print_stage("Stage 4d: Vector retrieval path selected")

        print_stage("Stage 5: Building augmented query")

        if route == "graph":
            augmented_query = f"""
You are a question answering system.

Answer in clear natural language (not JSON).

Use ONLY the graph relationships below as facts.

Graph knowledge:
{graph_context}

Question:
{user_query}

Rules:
- Answer in one or two sentences.
- Use entity names exactly as written.
- Do NOT output JSON.
- If unknown, say "I don't know".
"""

        else:
            augmented_query = f"""
You are a factual assistant.

Answer ONLY using the information below.
If the answer is not contained, say "I don't know".

Memory:
{memory_text}

Context:
{graph_context}

Question:
{user_query}
"""

        print_stage("Stage 6: Running LLM generation")

        try:
            if route == "graph":
                answer = graph_llm.complete(augmented_query).text
            else:
                response = query_engine.query(user_query)
                answer = str(response)
        except Exception as e:
            print("LLM error:", e)
            continue

        print_stage("Stage 7: LLM response received")

        print_stage("Stage 8: Storing answer in long-term memory")
        store = False

        if route == "graph":
            store = True
        else:
            if len(answer) > 60 and "i don't know" not in answer.lower():
                store = True

        if store:
            memory_agent.store_text(
                answer,
                metadata={"source": "assistant", "route": route}
            )

        print_stage("Stage 9: Memory storage completed")

        print("\nAssistant:\n")
        print(answer)


if __name__ == "__main__":
    main()
