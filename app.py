
import hashlib
from pathlib import Path

import streamlit as st

from src.pdf_processor import extract_pdf_pages
from src.text_chunker import chunk_pages
from src.embeddings import embed_texts, embed_query
from src.vector_store import VectorStore
from src.llm import generate_answer


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent

STORE_DIR = (
    PROJECT_ROOT / "data" / "documind_store"
)

st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# 2. LOAD OR INITIALIZE THE VECTOR STORE
# ==================================================

if "vector_store" not in st.session_state:

    vector_store = VectorStore()

    try:

        loaded = vector_store.load(STORE_DIR)

        if loaded:

            st.session_state.storage_message = (
                "Previously processed documents loaded successfully."
            )

        else:

            st.session_state.storage_message = (
                "No previously saved documents found."
            )

    except Exception as error:

        st.session_state.storage_message = (
            f"Could not load saved documents: {error}"
        )

    st.session_state.vector_store = vector_store


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# ==================================================
# RETRIEVAL AND CHUNKING SETTINGS
# ==================================================

with st.sidebar:

    st.header("⚙️ DocuMind Settings")

    st.subheader("Document Processing")

    chunk_size = st.number_input(
        "Chunk Size (characters)",
        min_value=300,
        max_value=3000,
        value=1000,
        step=100
    )

    overlap = st.number_input(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=299,
        value=150,
        step=25
    )

    st.divider()

    st.subheader("Retrieval Settings")

    top_k = st.slider(
        "Number of Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=4
    )

    min_score = st.slider(
        "Minimum Similarity Score",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05
    )


store = st.session_state.vector_store


# ==================================================
# 3. MAIN INTERFACE
# ==================================================

st.title("📚 DocuMind")

st.subheader(
    "AI-Powered Document Intelligence System"
)

st.write(
    """
    Upload your PDF documents, build your personal
    knowledge base and ask questions about them.

    Your processed document index is saved locally.
    """
)

st.caption(
    "Local embeddings + FAISS retrieval + Gemini generation"
)

st.divider()


# ==================================================
# 4. DOCUMENT UPLOAD AND PROCESSING
# ==================================================

st.header("Upload Documents")

uploaded_files = st.file_uploader(
    "Select PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


if st.button(
    "Process Selected Documents",
    disabled=not uploaded_files
):

    pending_documents = []

    skipped_count = 0

    processing_errors = []

    seen_in_batch = set()

    # ----------------------------------------------
    # A. EXTRACT PDF TEXT AND GENERATE CHUNKS
    # ----------------------------------------------

    with st.spinner("Processing uploaded PDFs..."):

        for uploaded_file in uploaded_files:

            pdf_bytes = uploaded_file.getvalue()

            # Generate unique ID using file contents.
            document_id = hashlib.sha256(
                pdf_bytes
            ).hexdigest()

            # Skip documents already in the database.
            if (
                store.has_document(document_id)
                or document_id in seen_in_batch
            ):

                skipped_count += 1
                continue

            seen_in_batch.add(document_id)

            try:

                pages = extract_pdf_pages(
                    pdf_bytes,
                    uploaded_file.name
                )

                # Attach document identity to every page.
                for page in pages:

                    page["document_id"] = document_id

                    document_chunks = chunk_pages(
                        pages,
                        chunk_size=int(chunk_size),
                        overlap=int(overlap)
                    )

                if not document_chunks:

                    raise ValueError(
                        "No selectable text found in this PDF."
                    )

                pending_documents.append({

                    "document_id": document_id,

                    "filename": uploaded_file.name,

                    "file_size": len(pdf_bytes),

                    "page_count": len(pages),

                    "chunks": document_chunks

                })

            except Exception as error:

                processing_errors.append(
                    f"{uploaded_file.name}: {error}"
                )

    # ----------------------------------------------
    # B. GENERATE EMBEDDINGS AND UPDATE FAISS
    # ----------------------------------------------

    total_new_chunks = sum(
        len(document["chunks"])
        for document in pending_documents
    )

    if pending_documents:

        try:

            all_new_chunks = []

            for document in pending_documents:

                all_new_chunks.extend(
                    document["chunks"]
                )

            with st.spinner(
                "Generating document embeddings..."
            ):

                embeddings = embed_texts(
                    [
                        chunk["text"]
                        for chunk in all_new_chunks
                    ]
                )

            with st.spinner(
                "Updating FAISS vector index..."
            ):

                store.add_chunks(
                    all_new_chunks,
                    embeddings
                )

                # Register each processed document.
                for document in pending_documents:

                    store.register_document(
                        document_id=document["document_id"],
                        filename=document["filename"],
                        page_count=document["page_count"],
                        chunk_count=len(document["chunks"]),
                        file_size=document["file_size"],
                        chunk_size=int(chunk_size),
                        overlap=int(overlap)
                    )

                # Save the complete updated store.
                store.save(STORE_DIR)
                st.session_state.storage_message = (
                "Documents are saved successfully on local storage."
                )

            st.session_state.chat_history = []

            st.success(
                f"Successfully processed "
                f"{len(pending_documents)} document(s) "
                f"and {total_new_chunks} new chunks."
            )

        except Exception as error:

            st.error(
                f"Unable to update document storage: {error}"
            )

    elif not processing_errors and skipped_count == 0:

        st.info("No new documents to process.")

    if skipped_count:

        st.info(
            f"Skipped {skipped_count} duplicate "
            "document(s)."
        )

    for error in processing_errors:

        st.error(error)


# ==================================================
# 5. DOCUMENT MANAGEMENT SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📁 My Documents")

    documents = store.get_documents()

    if not documents:

        st.info("No processed documents yet.")

    else:

        st.caption(
            f"{store.document_count()} document(s) stored"
        )

        for document in documents:

            with st.container(border=True):

                st.markdown(
                    f"**{document['filename']}**"
                )

                st.caption(
                    f"{document['page_count']} pages · "
                    f"{document['chunk_count']} chunks"
                )

                st.caption(
                    f"Chunk size: {document.get('chunk_size', 'Previous settings')} "
                    f"| Overlap: {document.get('overlap', 'Previous settings')}"
                )
                st.caption(
                    f"Added: {document['added_at'][:10]}"
                )

                if st.button(
                    "Remove Document",
                    key=f"remove_{document['document_id']}"
                ):

                    removed = store.remove_document(
                        document["document_id"]
                    )

                    if removed:

                        try:

                            store.save(STORE_DIR)
                            st.session_state.storage_message = (
                            "Document removed and storage updated successfully."
                        )
                            st.session_state.chat_history = []

                            st.success(
                                "Document removed successfully."
                            )

                            st.rerun()

                        except Exception as error:

                            st.error(
                                f"Could not save updated storage: {error}"
                            )


# ==================================================
# 6. STORAGE STATUS AND DASHBOARD
# ==================================================

with st.expander("Storage information"):

    st.write(
        st.session_state.storage_message
    )

    st.write(
        f"Storage location: `{STORE_DIR}`"
    )

    st.write(
        f"Stored vectors: {store.count()}"
    )

    st.write(
        f"Stored documents: {store.document_count()}"
    )


if store.count() > 0:

    st.divider()

    st.header("Document Dashboard")

    documents = store.get_documents()

    total_pages = sum(
        document["page_count"]
        for document in documents
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Documents",
        store.document_count()
    )

    col2.metric(
        "Total Pages",
        total_pages
    )

    col3.metric(
        "Total Chunks",
        store.count()
    )

    st.success(
        "Your documents are indexed and ready."
    )


# ==================================================
# 7. QUESTION AND ANSWER
# ==================================================

if store.count() > 0:

    st.divider()

    st.header("Ask Your Documents")

    with st.form("question_form"):

        question = st.text_input(
            "Enter your question",

            placeholder=(
                "Example: Explain virtualization."
            )
        )

        submitted = st.form_submit_button(
            "Ask DocuMind"
        )

    if submitted:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                # Generate question embedding.
                with st.spinner(
                    "Understanding your question..."
                ):

                    query_embedding = embed_query(
                        question
                    )

                # Search FAISS.

                # Step 2: Retrieve relevant chunks.

                with st.spinner("Searching documents..."):

                    retrieved_chunks = store.search(
                        query_embedding,
                        top_k=int(top_k),
                        min_score=float(min_score)
                    )


                # Step 3: Check retrieval results.

                if not retrieved_chunks:

                    st.warning(
                        "No sufficiently relevant information was found. "
                        "Try rephrasing your question or lowering "
                        "the similarity threshold."
                    )

                else:

                    # Step 4: Generate answer using Gemini.

                    with st.spinner(
                        "Gemini is generating your answer..."
                    ):

                        answer, sources = generate_answer(
                            question,
                            retrieved_chunks
                        )

                    st.session_state.chat_history.append({

                        "question": question,

                        "answer": answer,

                        "sources": sources

                    })

            except Exception as error:

                st.error(
                    f"Unable to answer your question: {error}"
                )


# ==================================================
# 8. DISPLAY CHAT HISTORY
# ==================================================

for chat in st.session_state.chat_history:

    st.divider()

    st.markdown(
        f"**You:** {chat['question']}"
    )

    st.markdown("### DocuMind's Answer")

    st.markdown(
        chat["answer"]
    )

    with st.expander(
        f"View Sources ({len(chat['sources'])})"
    ):

        for source in chat["sources"]:

            st.markdown(
                f"**[{source['id']}] "
                f"{source['source']} "
                f"— Page {source['page']}**"
            )

            st.caption(
                f"Similarity: {source['score']:.4f}"
            )

            st.write(
                source["text"]
            )

            st.divider()


# ==================================================
# 9. FOOTER
# ==================================================

st.divider()

st.caption(
    "DocuMind | AI-Powered Document Intelligence System"
)
