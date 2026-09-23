import faiss
import numpy as np


def create_vector_store(embeddings):
    """
    Create a FAISS index from document embeddings.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings).astype("float32"))

    return index


def search_vector_store(index, query_embedding, chunks, top_k=3):
    """
    Find the most relevant chunks for a query.
    """

    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = []

    for index_position in indices[0]:
        if index_position < len(chunks):
            results.append(chunks[index_position])

    return results