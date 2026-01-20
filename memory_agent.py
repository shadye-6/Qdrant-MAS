from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from uuid import uuid4
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
from PIL import Image


class MemoryAgent:
    def __init__(self, client: QdrantClient):
        self.client = client
        self.collection = "memory_collection"

        # Text embedding model
        self.text_embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Image embedding model (CLIP)
        self.image_embed_model = SentenceTransformer("clip-ViT-B-32")

        self.vector_size = 512  # CLIP vector size

        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config={"size": self.vector_size, "distance": "Cosine"}
        )

    def _pad_vector(self, vec):
        if len(vec) < self.vector_size:
            return vec + [0.0] * (self.vector_size - len(vec))
        return vec[:self.vector_size]

    # ---------- Text memory ----------

    def store_text(self, text, metadata=None):
        vec = self.text_embed_model.get_text_embedding(text)
        vec = self._pad_vector(vec)

        self._upsert(vec, {"type": "text", "text": text, **(metadata or {})})

    def retrieve_text(self, query, top_k=5):
        qvec = self.text_embed_model.get_text_embedding(query)
        qvec = self._pad_vector(qvec)

        return self.client.search(
            collection_name=self.collection,
            query_vector=qvec,
            limit=top_k
        )

    # ---------- Image memory ----------

    def store_image(self, image_path, metadata=None):
        image = Image.open(image_path).convert("RGB")
        vec = self.image_embed_model.encode(image).tolist()

        self._upsert(vec, {"type": "image", "image_path": image_path, **(metadata or {})})

    def retrieve_image(self, image_path, top_k=5):
        image = Image.open(image_path).convert("RGB")
        qvec = self.image_embed_model.encode(image).tolist()

        return self.client.search(
            collection_name=self.collection,
            query_vector=qvec,
            limit=top_k
        )

    # ---------- Core ----------

    def _upsert(self, vector, payload):
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def delete_all(self):
        self.client.delete_collection(self.collection)
