from ingest import load_documents, split_documents
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# Custom wrapper so LangChain can use sentence-transformers
class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


def create_vector_store():
    print("Loading documents...")
    documents = load_documents()

    print("Splitting into chunks...")
    chunks = split_documents(documents)

    print("Creating embeddings...")
    embedding_model = LocalEmbeddings()

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    vectorstore.save_local("embeddings")

    print("Vector database saved successfully!")


if __name__ == "__main__":
    create_vector_store()
