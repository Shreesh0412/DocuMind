
from pathlib import Path

from src.vector_store import VectorStore
from src.embeddings import embed_query


# Project storage location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

STORE_DIR = (
    PROJECT_ROOT / "data" / "documind_store"
)


def main():

    store = VectorStore()

    # Load saved FAISS database.
    loaded = store.load(STORE_DIR)

    if not loaded:

        print("No saved vector database found.")
        return

    print("\nDocuMind Retrieval Evaluator")

    print("Documents:", store.document_count())

    print("Total chunks:", store.count())

    print("\nType 'exit' to stop.")

    while True:

        question = input(
            "\nEnter your question: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            # Generate query embedding.
            query_embedding = embed_query(question)

            # Retrieve top candidates without threshold.
            results = store.search(
                query_embedding,
                top_k=5,
                min_score=None
            )

            print("\nRetrieved Results:")

            if not results:

                print("No results found.")
                continue

            for rank, result in enumerate(
                results,
                start=1
            ):

                print("\n--------------------------")

                print("Rank:", rank)

                print("Document:", result["source"])

                print("Page:", result["page"])

                print(
                    "Similarity:",
                    round(result["score"], 4)
                )

                print(
                    "Text:",
                    result["text"][:350]
                )

        except Exception as error:

            print("Retrieval error:", error)


if __name__ == "__main__":
    main()
