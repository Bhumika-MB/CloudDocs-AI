import streamlit as st
import tempfile
import os
import json
import urllib.request


# -----------------------------
# Cloudflare Worker
# -----------------------------

CLOUDFLARE_WORKER_URL = "https://clouddocs-ai-api.diyasharmadiya757.workers.dev"


def log_question_to_cloudflare(question):

    data = json.dumps({
        "question": question
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{CLOUDFLARE_WORKER_URL}/logs",
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status == 200

    except Exception as e:
        print("Cloudflare logging error:", e)
        return False
    
def get_query_history():

    try:
        request = urllib.request.Request(
            f"{CLOUDFLARE_WORKER_URL}/logs",
            headers={
                "User-Agent": "CloudDocs-AI"
            },
            method="GET"
        )

        with urllib.request.urlopen(request, timeout=5) as response:

            data = response.read().decode("utf-8")
            return json.loads(data)

    except Exception as e:

        st.error(f"Cloudflare history error: {e}")
        return []

from rag.pipeline import build_rag_pipeline


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="CloudDocs AI",
    page_icon="☁️",
    layout="wide"
)


# -----------------------------
# Custom styling
# -----------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.hero {
    padding: 2rem 0 1rem 0;
}

.hero h1 {
    font-size: 3rem;
    margin-bottom: 0.3rem;
}

.hero p {
    font-size: 1.15rem;
    color: #64748b;
}

.answer-box {
    padding: 1.5rem;
    border-radius: 14px;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    margin-top: 1rem;
}

.source-box {
    padding: 1rem;
    border-radius: 10px;
    background-color: #f1f5f9;
    margin-top: 0.5rem;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------

st.markdown("""
<div class="hero">

# ☁️ CloudDocs AI

### Ask questions about your documents using Retrieval-Augmented Generation

Upload a PDF and let CloudDocs AI retrieve the most relevant information
before generating an AI-powered answer.

</div>
""", unsafe_allow_html=True)


st.divider()


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    st.divider()

    st.markdown("""
    **How it works**

    1. 📄 Upload a document
    2. ✂️ Split it into chunks
    3. 🧠 Create embeddings
    4. 🔎 Search relevant information
    5. 🤖 Generate a grounded answer
    """)


# -----------------------------
# Main application
# -----------------------------

if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    question = st.text_input(
        "💬 What would you like to know?",
        placeholder="Example: What is the minimum CGPA required?"
    )

    if st.button(
        "🔍 Ask CloudDocs AI",
        type="primary",
        use_container_width=True
    ):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Reading document and generating answer..."):

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    pdf_path = temp_file.name

                try:

                    # Run RAG
                    result = build_rag_pipeline(
                        pdf_path,
                        question,
                        top_k=3
                    )
                    log_question_to_cloudflare(question)

                    # Answer
                    st.subheader("🤖 Answer")

                    st.markdown(
                        f"""
                        <div class="answer-box">
                        {result["answer"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Sources
                    st.subheader("📚 Sources")

                    for source in result["sources"]:

                        st.markdown(
                            f"""
                            <div class="source-box">
                            📄 <strong>Page {source["page"]}</strong>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                finally:

                    # Remove temporary PDF
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

else:

    st.info(
        "👈 Upload a PDF from the sidebar to start asking questions."
    )

# -----------------------------
# Query History
# -----------------------------

st.divider()

st.subheader("🕘 Query History")

history = get_query_history()

if history:

    for item in history:

        st.markdown(
            f"""
            <div class="source-box">
            💬 <strong>{item["question"]}</strong><br>
            🕐 {item["created_at"]}
            </div>
            """,
            unsafe_allow_html=True
        )

else:

    st.info("No previous questions yet.")    
    