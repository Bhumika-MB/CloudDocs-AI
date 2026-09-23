from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def create_chunks(pages, chunk_size=800, overlap=150):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "page": page["page"]
            })

            start += chunk_size - overlap

    return chunks