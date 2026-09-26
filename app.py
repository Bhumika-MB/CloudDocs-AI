import streamlit as st
import tempfile
import os
import json
import urllib.request
import urllib.error
import hashlib
import io

from pypdf import PdfReader

from rag.pipeline import build_rag_pipeline


# ============================================================
# CONFIGURATION
# ============================================================

CLOUDFLARE_WORKER_URL = (
    "https://clouddocs-ai-api.diyasharmadiya757.workers.dev"
)


# ============================================================
# CLOUDFARE API HELPERS
# ============================================================

def log_question_to_cloudflare(
    question,
    answer,
    document_name,
    document_id,
    sources
):
    """
    Store a question/answer in Cloudflare D1 through the Worker.
    """

    data = json.dumps({
        "question": question,
        "answer": answer,
        "document_name": document_name,
        "document_id": document_id,
        "sources": sources
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{CLOUDFLARE_WORKER_URL}/logs",
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "CloudDocs-AI"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response_body = response.read().decode("utf-8")

            if response.status == 200:
                return True

            print("Cloudflare POST response:", response_body)
            return False

    except Exception as e:
        print("Cloudflare logging error:", e)
        return False


def get_query_history():
    """
    Fetch query history from Cloudflare.
    """

    request = urllib.request.Request(
        f"{CLOUDFLARE_WORKER_URL}/logs",
        headers={
            "User-Agent": "CloudDocs-AI"
        },
        method="GET"
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:

            if response.status != 200:
                print(
                    "Cloudflare history status:",
                    response.status
                )
                return []

            data = response.read().decode("utf-8")

            if not data.strip():
                return []

            result = json.loads(data)

            if isinstance(result, list):
                return result

            return []

    except Exception as e:
        print("Cloudflare history error:", e)
        return []


def delete_query_from_cloudflare(query_id):
    """
    Delete one history entry.
    """

    request = urllib.request.Request(
        f"{CLOUDFLARE_WORKER_URL}/logs/{query_id}",
        headers={
            "User-Agent": "CloudDocs-AI"
        },
        method="DELETE"
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:

            if response.status == 200:
                return True

            return False

    except Exception as e:
        print("Cloudflare delete error:", e)
        return False


def clear_query_history():
    """
    Delete all history entries.
    """

    request = urllib.request.Request(
        f"{CLOUDFLARE_WORKER_URL}/logs",
        headers={
            "User-Agent": "CloudDocs-AI"
        },
        method="DELETE"
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:

            if response.status == 200:
                return True

            return False

    except Exception as e:
        print("Cloudflare clear history error:", e)
        return False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CloudDocs AI",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "history_refresh" not in st.session_state:
    st.session_state.history_refresh = 0


# ============================================================
# CUSTOM CSS
#
# IMPORTANT:
# Only CSS is injected here.
# The actual application UI uses Streamlit components,
# so raw HTML cannot accidentally appear on the page.
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #e5e7eb;
}

.brand-title {
    font-size: 1.35rem;
    font-weight: 750;
    margin-bottom: 0.2rem;
}

.brand-subtitle {
    color: #64748b;
    font-size: 0.88rem;
    line-height: 1.5;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    line-height: 1.6;
    max-width: 850px;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 750;
}

.answer-card {
    padding: 1.25rem;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    background: white;
    line-height: 1.7;
}

.source-card {
    padding: 0.9rem 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    background: #f8fafc;
    text-align: center;
}

.history-card {
    padding: 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    background: white;
}

.small-muted {
    color: #64748b;
    font-size: 0.85rem;
}

.stat-label {
    color: #64748b;
    font-size: 0.85rem;
}

.stat-value {
    font-size: 1.2rem;
    font-weight: 700;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ☁️ CloudDocs AI")

    st.caption(
        "AI-powered document question answering"
    )

    st.divider()

    st.markdown("### 📄 Your document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a PDF document to ask questions about it."
    )

    if uploaded_file:

        file_size_kb = uploaded_file.size / 1024

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        st.caption(
            f"{file_size_kb:.1f} KB"
        )

    st.divider()

    st.markdown("### ⚡ RAG Pipeline")

    pipeline_steps = [
        "📄 Upload document",
        "✂️ Split into chunks",
        "🧠 Create embeddings",
        "🔎 Retrieve relevant context",
        "🤖 Generate grounded answer"
    ]

    for i, step in enumerate(pipeline_steps, start=1):

        st.markdown(
            f"**{i}.** {step}"
        )

    st.divider()

    st.caption(
        "CloudDocs AI • Retrieval-Augmented Generation"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">☁️ CloudDocs AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    "Chat with your documents. Upload a PDF, ask questions "
    "in natural language, and get AI-generated answers "
    "grounded in the information inside your document."
    "</div>",
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# NO DOCUMENT STATE
# ============================================================

if not uploaded_file:

    st.info(
        "👈 Upload a PDF from the sidebar to start asking questions."
    )

    st.markdown("### How it works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄")
        st.markdown("**1. Upload**")
        st.caption(
            "Upload the PDF you want to search."
        )

    with col2:
        st.markdown("### 🔎")
        st.markdown("**2. Retrieve**")
        st.caption(
            "Relevant document chunks are retrieved."
        )

    with col3:
        st.markdown("### 🤖")
        st.markdown("**3. Ask**")
        st.caption(
            "Get an answer grounded in your document."
        )

else:

    # ========================================================
    # READ PDF INFORMATION
    # ========================================================

    pdf_bytes = uploaded_file.getvalue()

    document_id = hashlib.sha256(
        pdf_bytes
    ).hexdigest()[:16]

    try:

        pdf_reader = PdfReader(
            io.BytesIO(pdf_bytes)
        )

        page_count = len(
            pdf_reader.pages
        )

    except Exception:
        page_count = 0


    # ========================================================
    # DOCUMENT SUMMARY
    # ========================================================

    st.divider()

    doc_col1, doc_col2, doc_col3 = st.columns(3)

    with doc_col1:

        st.caption("📄 Document")

        st.markdown(
            f"### {uploaded_file.name}"
        )

    with doc_col2:

        st.caption("📑 Pages")

        if page_count > 0:
            st.markdown(
                f"### {page_count}"
            )
        else:
            st.markdown("### —")

    with doc_col3:

        st.caption("🔐 Document ID")

        st.markdown(
            f"`{document_id}`"
        )


    # ========================================================
    # QUESTION AREA
    # ========================================================

    st.divider()

    st.markdown(
        "## 💬 Ask your document"
    )

    question = st.text_input(
        "Question",
        placeholder=(
            "Example: What is the minimum CGPA required?"
        ),
        label_visibility="collapsed"
    )

    ask_button = st.button(
        "🔍  Ask CloudDocs AI",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # ASK QUESTION
    # ========================================================

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "🔎 Searching the document and generating your answer..."
            ):

                pdf_path = None

                try:

                    # ----------------------------------------
                    # Save temporary PDF
                    # ----------------------------------------

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(
                            pdf_bytes
                        )

                        pdf_path = temp_file.name


                    # ----------------------------------------
                    # Run RAG pipeline
                    # ----------------------------------------

                    result = build_rag_pipeline(
                        pdf_path,
                        question,
                        top_k=3
                    )


                    # ----------------------------------------
                    # Extract answer
                    # ----------------------------------------

                    answer = result.get(
                        "answer",
                        "No answer was generated."
                    )


                    # ----------------------------------------
                    # Extract source pages
                    # ----------------------------------------

                    raw_sources = result.get(
                        "sources",
                        []
                    )

                    source_pages = []

                    for source in raw_sources:

                        if not isinstance(
                            source,
                            dict
                        ):
                            continue

                        page = source.get(
                            "page"
                        )

                        if page is None:
                            continue

                        try:
                            page = int(page)
                        except:
                            pass

                        if page not in source_pages:
                            source_pages.append(
                                page
                            )

                    # Sort numeric pages properly
                    try:
                        source_pages = sorted(
                            source_pages,
                            key=lambda x: int(x)
                        )
                    except:
                        source_pages = sorted(
                            source_pages,
                            key=str
                        )


                    # ----------------------------------------
                    # Save to session state
                    # ----------------------------------------

                    st.session_state.last_answer = answer

                    st.session_state.last_sources = (
                        source_pages
                    )


                    # ----------------------------------------
                    # Prepare source string
                    # ----------------------------------------

                    sources_text = ", ".join(
                        str(page)
                        for page in source_pages
                    )


                    # ----------------------------------------
                    # Store history in Cloudflare
                    # ----------------------------------------

                    saved = log_question_to_cloudflare(
                        question=question.strip(),
                        answer=answer,
                        document_name=uploaded_file.name,
                        document_id=document_id,
                        sources=sources_text
                    )


                    if saved:

                        st.session_state.history_refresh += 1

                    else:

                        st.warning(
                            "The answer was generated, "
                            "but the query could not be saved "
                            "to history."
                        )


                except Exception as e:

                    st.error(
                        f"Something went wrong: {e}"
                    )

                finally:

                    if (
                        pdf_path
                        and os.path.exists(pdf_path)
                    ):
                        os.remove(pdf_path)


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    if st.session_state.last_answer:

        st.divider()

        st.markdown(
            "## 🤖 Answer"
        )

        with st.container(border=True):

            st.markdown(
                st.session_state.last_answer
            )


        # ====================================================
        # SOURCES
        # ====================================================

        st.write("")

        st.markdown(
            "## 📚 Sources"
        )

        sources = (
            st.session_state.last_sources
        )

        if sources:

            source_columns = st.columns(
                min(len(sources), 4)
            )

            for index, page in enumerate(
                sources
            ):

                with source_columns[
                    index % len(source_columns)
                ]:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"📄 **Page {page}**"
                        )

        else:

            st.info(
                "No source pages were returned."
            )


# ============================================================
# QUERY HISTORY
# ============================================================

st.divider()

history_title_col, history_button_col = st.columns(
    [5, 1]
)

with history_title_col:

    st.markdown(
        "## 🕘 Query History"
    )

with history_button_col:

    history = get_query_history()

    if history:

        clear_clicked = st.button(
            "🗑️ Clear All",
            use_container_width=True
        )

        if clear_clicked:

            with st.spinner(
                "Clearing history..."
            ):

                success = clear_query_history()

            if success:

                st.success(
                    "Query history cleared."
                )

                st.rerun()

            else:

                st.error(
                    "Could not clear query history."
                )


# ============================================================
# DISPLAY HISTORY
# ============================================================

if history:

    for item in history:

        query_id = item.get(
            "id"
        )

        question_text = item.get(
            "question",
            "Unknown question"
        )

        answer_text = item.get(
            "answer",
            ""
        )

        document_name = item.get(
            "document_name",
            ""
        )

        sources = item.get(
            "sources",
            []
        )

        created_at = item.get(
            "created_at",
            ""
        )


        # --------------------------------------------
        # Normalize sources
        # --------------------------------------------

        if isinstance(
            sources,
            str
        ):

            sources = [
                x.strip()
                for x in sources.split(",")
                if x.strip()
            ]

        elif not isinstance(
            sources,
            list
        ):

            sources = []


        # --------------------------------------------
        # History card
        # --------------------------------------------

        with st.container(
            border=True
        ):

            history_col1, history_col2 = st.columns(
                [6, 1]
            )

            with history_col1:

                st.markdown(
                    f"### 💬 {question_text}"
                )

                if created_at:

                    st.caption(
                        f"🕐 {created_at}"
                    )

            with history_col2:

                if query_id is not None:

                    delete_clicked = st.button(
                        "🗑️",
                        key=f"delete_{query_id}",
                        help="Delete this query"
                    )

                    if delete_clicked:

                        success = (
                            delete_query_from_cloudflare(
                                query_id
                            )
                        )

                        if success:

                            st.rerun()

                        else:

                            st.error(
                                "Delete failed."
                            )


            with st.expander(
                "View details"
            ):

                if answer_text:

                    st.markdown(
                        "#### 🤖 Answer"
                    )

                    st.markdown(
                        answer_text
                    )

                if document_name:

                    st.markdown(
                        f"**📄 Document:** "
                        f"`{document_name}`"
                    )

                if sources:

                    st.markdown(
                        "**📚 Sources:** "
                        + ", ".join(
                            f"Page {page}"
                            for page in sources
                        )
                    )

else:

    st.info(
        "No previous questions yet. "
        "Ask a question to create your first history entry."
    )


# ============================================================
# FOOTER
# ============================================================

st.write("")
st.divider()

st.caption(
    "☁️ CloudDocs AI • "
    "Retrieval-Augmented Generation • "
    "Powered by Streamlit"
)