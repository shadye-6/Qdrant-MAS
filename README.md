# Multimodal Neuro-Symbolic RAG Agent

A local-first, multimodal Retrieval-Augmented Generation (RAG) system combining **vector search**, **knowledge graphs**, and **long-term memory** with traceable reasoning.

This project ingests documents (text, PDF, PPT, images), extracts structured entities and relations into **Neo4j**, embeds content into **Qdrant**, stores conversational memory with decay-ready architecture, and dynamically routes user queries to either vector retrieval or graph reasoning.

The system is designed for research and experimentation in **neuro-symbolic AI**, **hybrid retrieval**, and **transparent RAG pipelines**.

----------

## Key Features

-   Multimodal ingestion: TXT, PDF, PPT/PPTX, images
-   Knowledge graph extraction using local LLM (llama.cpp)
-   Vector search using Qdrant + SentenceTransformers
-   Graph storage and querying using Neo4j
-   Long-term memory agent (text + image embeddings)
-   Query router (vector vs graph reasoning)
-   Local LLM inference (HuggingFace or llama.cpp)
-   Traceable chunk-level knowledge linking
-   Modular agent architecture

----------

## System Architecture (High Level)

```
Documents
  |
  v
Document Processor
  +--> Vector Embeddings (Qdrant)
  +--> Graph Extraction (LLM) --> Neo4j Knowledge Graph
  +--> Chunk Metadata

User Query
  |
  v
OrchestratorAgent
  |-- vector --> Qdrant retrieval
  |-- graph  --> Neo4j relationship query
  v
LLM Generation
  v
MemoryAgent (Qdrant long-term memory)

```

----------

## Repository Structure

```
.
├── agents/
│   ├── graph_extractor.py      # LLM → entities + relations
│   ├── graph_store.py          # Neo4j interface
│   └── orchestrator.py         # Query routing logic
├── document_processors.py      # Multimodal ingestion
├── llm_llama_cpp.py            # llama.cpp wrapper
├── memory_agent.py             # Long-term memory with Qdrant
├── rag_terminal.py             # Main CLI application
├── requirements.txt
└── README.md

```

----------

## Requirements

### System

-   Python 3.9+
-   Neo4j 5.x (running locally)
-   LibreOffice (for PPT → PDF conversion)
-   CUDA (optional, for GPU acceleration)
-   llama.cpp compatible GGUF model (e.g., Mistral)

### Python Libraries

Installed from `requirements.txt`, including:

-   llama-index
-   qdrant-client
-   neo4j
-   sentence-transformers
-   transformers
-   torch
-   pillow
-   pymupdf (fitz)
-   python-pptx
-   llama-cpp-python

----------

## Installation

### 1. Clone repository

```bash
git clone <repo_url>
cd <repo_name>

```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

----------

## External Services Setup

### Neo4j

Start Neo4j locally:

```bash
docker run -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5

```

Default credentials used:

```
URI: bolt://localhost:7687
User: neo4j
Pass: password

```

You can change these in:

```
agents/graph_store.py

```

----------

### Qdrant

Local embedded storage is used automatically:

```
./qdrant_data/

```

No external server required.

----------

### llama.cpp Model

Download a GGUF model, e.g.:

-   Mistral 7B Instruct
-   Qwen 2.5
-   Llama 2

Place it in:

```
models/mistral.gguf

```

Or update:

```
initialize_llm()

```

in `rag_terminal.py`.

----------

## Running the System

```bash
python rag_terminal.py

```

You will be prompted:

```
Enter path to documents folder:

```

Provide a folder containing:

-   .txt
-   .pdf
-   .ppt / .pptx
-   .png / .jpg

----------

## Example Workflow

1.  System loads documents
2.  Splits text into chunks
3.  Extracts entities + relations using local LLM
4.  Stores graph in Neo4j
5.  Stores embeddings in Qdrant
6.  Waits for user queries

Example query:

```
How does component A affect system B?

```

Router selects graph path → Neo4j is queried.

Example:

```
Explain the design of the memory system

```

Router selects vector path → semantic retrieval.

----------

## Query Routing Logic

Defined in:

```
agents/orchestrator.py

```

Graph keywords:

```
relationship
connect
cause
impact
depend
affect
between
related
how does
why does

```

----------

## Knowledge Graph Format

LLM extraction format:

```json
{
  "entities": [
    { "name": "EntityA", "type": "concept" }
  ],
  "relations": [
    { "source": "EntityA", "target": "EntityB", "relation": "CAUSES" }
  ]
}

```

Stored as:

Nodes:

```
(:Entity {id, name, type})

```

Relations:

```
[:RELATION {type}]

```

Chunk links:

```
(:Chunk)-[:MENTIONS]->(:Entity)

```

----------

## Long-Term Memory

Implemented in `memory_agent.py`.

Features:

-   Text memory embedding
-   Image memory embedding (CLIP)
-   Stored in Qdrant
-   Retrieved every query

Can be cleared:

```
clear_memory

```

----------

## Configuration Points

Change LLM (Graph Extraction):

```
initialize_llm()

```

Change LLM (Answer generation):

```
initialize_hf_llm()

```

Chunk size:

```
SentenceSplitter(chunk_size=256, chunk_overlap=50)

```

Graph DB credentials:

```
GraphStore(...)

```

----------

## Limitations

-   Graph extraction depends on LLM compliance
-   No automatic graph schema evolution
-   No summarization or memory consolidation (by design)
-   Basic router heuristics
-   No web interface (CLI only)

----------

## Roadmap Ideas

-   Confidence scoring for relations
-   Memory decay scheduler
-   Graph embedding hybrid search
-   UI dashboard
-   Streaming generation
-   Tool-calling agents
-   Provenance tracing per answer token

----------
