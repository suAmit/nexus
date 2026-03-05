import os
import sqlite3
from datetime import datetime

import chromadb


class Database_Setup:
    def __init__(self):
        # 1. Initialize SQLite for Metadata/Logs
        self.db_path = "data/nexus_logs.db"
        self._init_sqlite()

        # 2. Initialize ChromaDB for Symmetric Semantic Memory
        self.chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        self.memory_collection = self.chroma_client.get_or_create_collection(
            name="semantic_memory",
            metadata={"hnsw:space": "cosine"},
        )

    def _init_sqlite(self):
        """Creates/Updates the logs table without process_logs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Removed 'process_logs' column
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                prompt TEXT,
                response TEXT,
                tier TEXT,
                cost REAL,
                latency REAL,
                category TEXT,
                model_version TEXT
            )
        """)
        conn.commit()
        conn.close()

    def check_cache(self, prompt, threshold=0.92):
        """L1 Cache: Looks for a nearly identical prompt."""
        results = self.memory_collection.query(query_texts=[prompt], n_results=1)

        if results["ids"] and results["distances"][0]:
            distance = results["distances"][0][0]
            if distance < (1 - threshold):
                return results["metadatas"][0][0].get("response")
        return None

    def get_context(self, prompt, n_results=3, threshold=0.70):
        """L2 Memory: Retrieves relevant past interactions."""
        results = self.memory_collection.query(
            query_texts=[prompt], n_results=n_results
        )
        context_parts = []

        if results["ids"] and results["distances"][0]:
            for i, distance in enumerate(results["distances"][0]):
                if distance < (1 - threshold):
                    prev_p = results["documents"][0][i]
                    prev_r = results["metadatas"][0][i].get("response")
                    context_parts.append(f"Past Query: {prev_p}\nPast Answer: {prev_r}")

        return "\n---\n".join(context_parts) if context_parts else ""

    def save_interaction(
        self, prompt, response, tier, cost=0.0, latency=0.0, logs="", model=""
    ):
        """Saves interaction symmetrically to Vector Store and SQL."""
        timestamp = datetime.now().isoformat()
        interaction_id = f"id_{datetime.now().timestamp()}"

        # Vector Store
        self.memory_collection.add(
            documents=[prompt],
            metadatas=[{"response": response, "tier": tier, "timestamp": timestamp}],
            ids=[interaction_id],
        )

        # SQL Store: Removed process_logs from INSERT
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO interaction_logs 
                (timestamp, prompt, response, tier, cost, latency, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (timestamp, prompt, response, tier, cost, latency, model),
            )
            conn.commit()
        except sqlite3.OperationalError:
            # If the DB still has the old schema, this prevents a crash
            # Better to manually delete the .db file once to reset schema
            pass
        finally:
            conn.close()
