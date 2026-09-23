from rag.ingestion import extract_text_from_pdf, create_chunks
from rag.embeddings import create_embeddings
from rag.retrieval import create_vector_store, search_vector_store
from rag.generation import generate_answer


def build_rag_pipeline(pdf_path, question, top_k=3):
    """
    Run the complete RAG pipeline for a PDF and question.
    """

    # 1. Extract PDF text
    pages = extract_text_from_pdf(pdf_path)

    # 2. Create chunks
    chunks = create_chunks(pages)

    # 3. Create embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    # 4. Create FAISS vector store
    index = create_vector_store(embeddings)

    # 5. Embed the question
    query_embedding = create_embeddings([question])[0]

    # 6. Retrieve relevant chunks
    results = search_vector_store(
        index,
        query_embedding,
        chunks,
        top_k=top_k
    )

    # 7. Build context
    context = "\n\n".join(
        f"Page {result['page']}:\n{result['text']}"
        for result in results
    )

    # 8. Generate answer
    answer = generate_answer(
        question,
        context
    )

    return {
        "answer": answer,
        "sources": results
    }