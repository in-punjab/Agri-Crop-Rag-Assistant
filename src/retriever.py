from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

# Same embedding class used earlier
class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


def load_vector_store():
    embedding_model = LocalEmbeddings()

    vectorstore = FAISS.load_local(
        "embeddings",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def ask_question(query):
    vectorstore = load_vector_store()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    results = retriever.invoke(query)


    print("\nTop Retrieved Knowledge:\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(doc.page_content)
        print("-" * 40)


if __name__ == "__main__":
    user_query = input("Ask your agriculture question: ")
    ask_question(user_query)
