import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Database.session import get_connection
from Config.model import get_embeddings
import json
import numpy as np


DOCS_PATH = "RAG/Docs"


def load_documents():
    documents = []

    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_PATH, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "filename": filename,
                "department": filename.replace(".txt", ""),
                "content": text
            })

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])

        for chunk in chunks:
            all_chunks.append({
                "department": doc["department"],
                "source_file": doc["filename"],
                "chunk_text": chunk
            })

    return all_chunks

def store_chunks(chunks):
    conn = get_connection()
    cur = conn.cursor()

    embeddings = get_embeddings()

    for chunk in chunks:
        vector = embeddings.embed_query(chunk["chunk_text"])
        vector_json = json.dumps(vector)

        cur.execute("""
            INSERT INTO rag_documents (department, chunk_text, embedding, source_file)
            VALUES (%s, %s, %s::jsonb, %s)
        """, (
            chunk["department"],
            chunk["chunk_text"],
            vector_json,
            chunk["source_file"]
        ))

    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)

    print(f"Loaded {len(docs)} documents")
    print(f"Generated {len(chunks)} chunks")

    store_chunks(chunks)

    print("Chunks embedded and stored successfully.")