"""
atomic_embedder.py
Atomic Q&A embedder for textbooks – minimizes LLM token usage.
"""

import hashlib
from typing import List, Dict, Tuple
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


class AtomicCurriculumEmbedder:
    def __init__(self, persist_directory: str = "./curriculum_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="atomic_curriculum",
            embedding_function=self.ef
        )
        self.cache: Dict[str, str] = {}

    def embed_curriculum(self, qa_pairs: List[Dict[str, str]]) -> int:
        for pair in qa_pairs:
            q = pair["question"].strip()
            a = pair["answer"].strip()
            doc_id = hashlib.md5(q.encode()).hexdigest()
            self.collection.add(
                documents=[q],
                metadatas=[{"answer": a, "tokens": len(a.split())}],
                ids=[doc_id]
            )
            self.cache[q.lower()] = a
        return len(qa_pairs)

    def ask(self, user_question: str, similarity_threshold: float = 0.7) -> Tuple[str, str]:
        normalized = user_question.lower().strip()
        if normalized in self.cache:
            return self.cache[normalized], "exact cache"

        results = self.collection.query(query_texts=[user_question], n_results=1)
        if not results['ids'][0]:
            return "No relevant answer found.", "no match"

        distance = results['distances'][0][0]
        if distance < similarity_threshold:
            answer = results['metadatas'][0][0]['answer']
            return answer, "vector DB"
        else:
            return "No confident match. Please rephrase.", "low confidence"

    def clear_cache(self):
        self.cache.clear()

    def delete_collection(self):
        self.client.delete_collection("atomic_curriculum")
        self.cache.clear()


if __name__ == "__main__":
    # Quick self-test
    test_qa = [
        {"question": "What is the derivative of x^2?", "answer": "2x"},
        {"question": "Area of a circle", "answer": "πr²"},
    ]
    embedder = AtomicCurriculumEmbedder()
    embedder.embed_curriculum(test_qa)
    print(embedder.ask("derivative of x squared"))  # Should print ('2x', 'exact cache')
    print(embedder.ask("circle area"))              # Should print ('πr²', 'vector DB')
