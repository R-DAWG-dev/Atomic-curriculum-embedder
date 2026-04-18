"""
Engine 1 — Database Presence & Health Check
Confirms the ChromaDB curriculum database exists and reports its status
before any other engine attempts to use it.
"""

import os
import sys

DB_PATH = "./curriculum_db"
COLLECTION_NAME = "paeds_curriculum"


def check_database() -> dict:
    report = {
        "db_path_exists": False,
        "collection_exists": False,
        "record_count": 0,
        "status": "NOT READY",
    }

    if not os.path.exists(DB_PATH):
        print(f"[FAIL] Database directory not found: {DB_PATH}")
        print("       Run the loader with your question banks first.")
        return report

    report["db_path_exists"] = True

    try:
        import chromadb

        client = chromadb.PersistentClient(path=DB_PATH)
        existing = [c.name for c in client.list_collections()]

        if COLLECTION_NAME not in existing:
            print(f"[FAIL] Collection '{COLLECTION_NAME}' not found.")
            print(f"       Collections present: {existing or 'none'}")
            return report

        report["collection_exists"] = True
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        report["record_count"] = count

        if count == 0:
            print("[WARN] Collection exists but is empty — load question banks first.")
            report["status"] = "EMPTY"
        else:
            print(f"[OK]   Database ready — {count:,} Q&A pairs loaded.")
            report["status"] = "READY"

    except ImportError:
        print("[ERROR] chromadb is not installed. Run: pip install chromadb")
        report["status"] = "ERROR"
    except Exception as exc:
        print(f"[ERROR] Database check failed: {exc}")
        report["status"] = "ERROR"

    return report


def load_question_bank(qa_pairs: list, reset: bool = False) -> int:
    """
    Load a list of {"question": ..., "answer": ...} dicts into the database.
    Set reset=True to wipe the collection before loading.
    Returns the number of records now in the collection.
    """
    import hashlib
    import chromadb
    from chromadb.utils import embedding_functions

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_PATH)

    if reset and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
        print("[INFO] Existing collection cleared.")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=ef
    )

    ids, docs, metas = [], [], []
    for pair in qa_pairs:
        q = pair["question"].strip()
        a = pair["answer"].strip()
        doc_id = hashlib.md5(q.encode()).hexdigest()
        ids.append(doc_id)
        docs.append(q)
        metas.append({"answer": a})

    if ids:
        collection.upsert(documents=docs, metadatas=metas, ids=ids)

    total = collection.count()
    print(f"[OK]   Loaded {len(ids)} records. Total in DB: {total:,}")
    return total


if __name__ == "__main__":
    result = check_database()
    if result["status"] not in ("READY",):
        sys.exit(1)
