from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import get_settings
from src.logging_config import configure_logging
from src.upload_validation import validate_pdf_upload_batch
from src.advanced_features import (
    summarize_document,
    generate_mcqs,
    compare_documents,
)
from src.pdf_processor import extract_pdf_pages
from src.text_chunker import chunk_pages
from src.embeddings import embed_texts, embed_query
from src.vector_store import VectorStore
from src.llm import generate_answer

SETTINGS = get_settings()
configure_logging(SETTINGS.log_level)
logger = logging.getLogger("documind.app")

PROJECT_ROOT = Path(__file__).resolve().parent
STORE_DIR = PROJECT_ROOT / "data" / "documind_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1280px;
        padding-top: 1.8rem;
        padding-bottom: 2.8rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }

    .documind-brand {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.06rem;
        margin-bottom: 0;
    }

    .documind-tagline {
        color: #8b929d;
        font-size: 0.78rem;
        margin-top: -0.25rem;
    }

    .documind-hero {
        padding: 1.5rem;
        border: 1px solid rgba(100, 140, 200, 0.20);
        border-radius: 1rem;
        background: linear-gradient(
            115deg,
            rgba(65, 125, 220, 0.11),
            rgba(70, 170, 155, 0.07)
        );
        margin-bottom: 1rem;
    }

    .documind-hero h2 {
        margin: 0 0 0.45rem 0;
        letter-spacing: -0.04rem;
    }

    .documind-hero p {
        margin: 0;
        color: #8b929d;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 0.8rem;
        padding: 0.8rem 1rem;
        background: rgba(128, 128, 128, 0.035);
    }

    div[data-testid="stChatMessage"] {
        border-radius: 0.85rem;
    }

    div[data-testid="stSegmentedControl"] {
        margin: 0.35rem 0 1.1rem 0;
    }

    div[data-testid="stSegmentedControl"] button {
        min-height: 2.7rem;
        font-weight: 600;
    }

    .documind-section-caption {
        color: #8b929d;
        margin-top: -0.35rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def initialize_session_state() -> None:
    """Initialize the persistent vector store and chat state."""

    if "vector_store" not in st.session_state:
        vector_store = VectorStore(
            embedding_model=EMBEDDING_MODEL
        )

        storage_load_failed = False
        try:
            loaded = vector_store.load(STORE_DIR)
            if loaded:
                storage_message = "Previously saved documents were loaded."
            else:
                storage_message = "Your document library is currently empty."
        except Exception:
            logger.exception("Saved document storage could not be loaded.")
            storage_load_failed = True
            storage_message = (
                "Saved storage could not be loaded. Uploads and removal are paused "
                "to protect existing files. Check the application logs and restore "
                "a valid backup before restarting DocuMind."
            )

        st.session_state.vector_store = vector_store
        st.session_state.storage_message = storage_message
        st.session_state.storage_load_failed = storage_load_failed

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "summary_result" not in st.session_state:
        st.session_state.summary_result = None

    if "mcq_result" not in st.session_state:
        st.session_state.mcq_result = None

    if "mcq_attempt" not in st.session_state:
        st.session_state.mcq_attempt = None

    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None

    if "comparison_document_ids" not in st.session_state:
        st.session_state.comparison_document_ids = []

    if "selected_document_ids" not in st.session_state:
        st.session_state.selected_document_ids = list(
            st.session_state.vector_store.documents.keys()
        )

initialize_session_state()
store: VectorStore = st.session_state.vector_store

def process_uploaded_documents(
    uploaded_files: list[Any],
    chunk_size: int,
    overlap: int,
) -> tuple[int, int, list[str], int]:
    """Extract, chunk, embed, index, and persist newly uploaded PDFs."""
    global store

    if SETTINGS.demo_mode:
        raise RuntimeError(
            "Document uploads are disabled while DocuMind demo mode is enabled."
        )
    if st.session_state.get("storage_load_failed", False):
        raise RuntimeError(
            "Uploads are paused because the saved library failed to load. "
            "Restore a valid backup and restart DocuMind first."
        )

    file_sizes: list[tuple[str, int]] = []
    for uploaded_file in uploaded_files:
        uploaded_size = getattr(uploaded_file, "size", None)
        if uploaded_size is None:
            uploaded_size = len(uploaded_file.getvalue())
        file_sizes.append((uploaded_file.name, int(uploaded_size)))

    validate_pdf_upload_batch(
        file_sizes,
        max_files=SETTINGS.max_batch_files,
        max_file_size_bytes=SETTINGS.max_pdf_file_size_bytes,
        max_batch_size_bytes=SETTINGS.max_batch_size_bytes,
    )

    pending_documents: list[dict] = []
    processing_errors: list[str] = []
    seen_in_batch: set[str] = set()
    skipped_count = 0

    for uploaded_file, (_, expected_size) in zip(uploaded_files, file_sizes):
        pdf_bytes = uploaded_file.getvalue()
        if len(pdf_bytes) != expected_size:
            processing_errors.append(
                f"{uploaded_file.name}: uploaded file size changed during processing."
            )
            continue

        document_id = hashlib.sha256(pdf_bytes).hexdigest()

        if store.has_document(document_id) or document_id in seen_in_batch:
            skipped_count += 1
            continue
        seen_in_batch.add(document_id)

        try:
            pages = extract_pdf_pages(
                pdf_bytes,
                uploaded_file.name,
                max_pages=SETTINGS.max_pdf_pages,
            )

            for page in pages:
                page["document_id"] = document_id

            document_chunks = chunk_pages(
                pages,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            if not document_chunks:
                raise ValueError("No selectable text was found in this PDF.")

            existing_pending_chunks = sum(
                len(document["chunks"]) for document in pending_documents
            )
            if (
                existing_pending_chunks + len(document_chunks)
                > SETTINGS.max_chunks_per_batch
            ):
                raise ValueError(
                    "The selected PDFs generate too many text chunks. "
                    f"The batch limit is {SETTINGS.max_chunks_per_batch} chunks."
                )

            pending_documents.append(
                {
                    "document_id": document_id,
                    "filename": Path(uploaded_file.name).name,
                    "file_size": len(pdf_bytes),
                    "page_count": len(pages),
                    "chunks": document_chunks,
                }
            )
        except Exception as error:
            logger.warning(
                "PDF processing failed for %s (%s).",
                Path(uploaded_file.name).name,
                type(error).__name__,
            )
            processing_errors.append(f"{uploaded_file.name}: {error}")

    if not pending_documents:
        return 0, 0, processing_errors, skipped_count

    all_new_chunks: list[dict] = []
    for document in pending_documents:
        all_new_chunks.extend(document["chunks"])

    embeddings = embed_texts([chunk["text"] for chunk in all_new_chunks])
    updated_store = store.copy()
    updated_store.add_chunks(all_new_chunks, embeddings)

    for document in pending_documents:
        updated_store.register_document(
            document_id=document["document_id"],
            filename=document["filename"],
            page_count=document["page_count"],
            chunk_count=len(document["chunks"]),
            file_size=document["file_size"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

    # Publish the new immutable snapshot before replacing the live session state.
    updated_store.save(STORE_DIR)
    st.session_state.vector_store = updated_store
    store = updated_store
    logger.info(
        "Indexed %s PDF(s) and %s chunks.",
        len(pending_documents),
        len(all_new_chunks),
    )

    return (
        len(pending_documents),
        len(all_new_chunks),
        processing_errors,
        skipped_count,
    )

with st.sidebar:
    st.markdown(
        '<div class="documind-brand">📚 DocuMind</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="documind-tagline">'
        'AI-powered document intelligence'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Add documents")

    uploaded_files = []
    if SETTINGS.demo_mode:
        st.info(
            "Demo mode is enabled. The document library is read-only; "
            "uploads and document removal are disabled."
        )
    elif st.session_state.get("storage_load_failed", False):
        st.warning(
            "Uploads are paused because the saved library could not be loaded. "
            "Restore a valid backup and restart DocuMind before changing storage."
        )
    else:
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help=(
                f"Up to {SETTINGS.max_batch_files} PDFs per batch; "
                f"{SETTINGS.max_pdf_file_size_mb} MB per file; "
                f"{SETTINGS.max_batch_size_mb} MB combined."
            ),
            key="pdf_uploader",
        )
        st.caption(
            f"Limits: {SETTINGS.max_pdf_pages} pages per PDF and "
            f"{SETTINGS.max_chunks_per_batch} chunks per batch."
        )

    with st.expander("Processing settings", expanded=False):
        chunk_size = st.number_input(
            "Chunk size (characters)",
            min_value=300,
            max_value=3000,
            value=1000,
            step=100,
            key="chunk_size_setting",
        )

        overlap = st.number_input(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=int(chunk_size) - 1,
            value=min(150, int(chunk_size) - 1),
            step=25,
            key="chunk_overlap_setting",
        )

    process_clicked = st.button(
        "⚡ Process selected PDFs",
        type="primary",
        use_container_width=True,
        disabled=(
            SETTINGS.demo_mode
            or st.session_state.get("storage_load_failed", False)
            or not uploaded_files
        ),
    )

if process_clicked and uploaded_files:
    try:
        with st.status(
            "Processing your documents...",
            expanded=True,
        ) as status:
            status.write(
                "Extracting PDFs, generating embeddings, updating FAISS, "
                "and saving the document library..."
            )

            processed_count, chunk_count, errors, skipped_count = (
                process_uploaded_documents(
                    uploaded_files=uploaded_files,
                    chunk_size=int(chunk_size),
                    overlap=int(overlap),
                )
            )

            if processed_count:
                status.update(
                    label="Document processing completed.",
                    state="complete",
                    expanded=False,
                )

                st.session_state.selected_document_ids = list(
                    store.documents.keys()
                )
                st.session_state.chat_history = []
                st.session_state.summary_result = None
                st.session_state.mcq_result = None
                st.session_state.mcq_attempt = None
                st.session_state.comparison_result = None
                st.session_state.storage_message = (
                    "Documents and vector index saved successfully."
                )

                st.success(
                    f"Processed {processed_count} new document(s) "
                    f"and indexed {chunk_count} chunk(s)."
                )

            else:
                status.update(
                    label="No new documents were processed.",
                    state="complete",
                    expanded=False,
                )

        if skipped_count:
            st.info(
                f"Skipped {skipped_count} duplicate document(s)."
            )

        for error_message in errors:
            st.error(error_message)

        if processed_count:
            st.rerun()

        if not errors and not skipped_count:
            st.info("No new documents were selected.")

        if skipped_count and not errors and not processed_count:
            st.info("All selected PDFs are already in your library.")

    except Exception as error:
        logger.exception("Document batch processing failed.")
        st.error(f"Document processing failed: {error}")

with st.sidebar:
    st.divider()
    st.subheader("📁 Document library")

    documents = store.get_documents()

    if not documents:
        if st.session_state.get("storage_load_failed", False):
            st.error("The saved document library is unavailable pending recovery.")
        else:
            st.info("Your library is empty. Upload a PDF to begin.")
    else:
        st.caption(
            f"{store.document_count()} document(s) stored"
        )

        for document in documents:
            document_id = document["document_id"]

            with st.container(border=True):
                st.markdown(f"**{document['filename']}**")
                st.caption(
                    f"{document['page_count']} pages · "
                    f"{document['chunk_count']} chunks"
                )
                st.caption(
                    f"Chunk size: {document.get('chunk_size', '—')} · "
                    f"Overlap: {document.get('overlap', '—')}"
                )

                added_at = document.get("added_at", "")
                if added_at:
                    st.caption(f"Added: {added_at[:10]}")

                if SETTINGS.demo_mode:
                    st.caption("Demo mode: document removal is disabled.")
                elif st.session_state.get("storage_load_failed", False):
                    st.caption("Storage recovery required: document removal is paused.")
                else:
                    if st.button(
                        "Remove document",
                        key=f"remove_{document_id}",
                        use_container_width=True,
                    ):
                        try:
                            updated_store = store.copy()
                            if updated_store.remove_document(document_id):
                                updated_store.save(STORE_DIR)
                                st.session_state.vector_store = updated_store
                                store = updated_store

                                st.session_state.selected_document_ids = [
                                    current_id
                                    for current_id in (
                                        st.session_state.selected_document_ids
                                    )
                                    if current_id != document_id
                                ]

                                st.session_state.chat_history = []
                                current_summary = st.session_state.get(
                                    "summary_result"
                                )
                                if (
                                    current_summary
                                    and current_summary.get("document_id") == document_id
                                ):
                                    st.session_state.summary_result = None

                                current_mcq = st.session_state.get("mcq_result")
                                if (
                                    current_mcq
                                    and current_mcq.get("document_id") == document_id
                                ):
                                    st.session_state.mcq_result = None
                                    st.session_state.mcq_attempt = None

                                st.session_state.comparison_document_ids = [
                                    selected_id
                                    for selected_id in st.session_state.comparison_document_ids
                                    if selected_id != document_id
                                ]
                                current_comparison = st.session_state.get(
                                    "comparison_result"
                                )
                                if (
                                    current_comparison
                                    and document_id in current_comparison.get(
                                        "document_ids", []
                                    )
                                ):
                                    st.session_state.comparison_result = None

                                st.session_state.storage_message = (
                                    "Document removed and storage updated."
                                )
                                st.rerun()

                        except Exception as error:
                            st.error(
                                f"Could not remove document: {error}"
                            )

available_document_ids = list(store.documents.keys())
available_id_set = set(available_document_ids)

st.session_state.selected_document_ids = [
    document_id
    for document_id in st.session_state.selected_document_ids
    if document_id in available_id_set
]

with st.sidebar:
    st.divider()
    st.subheader("🔎 Search scope")

    selected_document_ids = st.multiselect(
        "Search within",
        options=available_document_ids,
        format_func=lambda document_id: (
            store.documents[document_id]["filename"]
        ),
        key="selected_document_ids",
        help="Choose which uploaded PDFs DocuMind should search.",
    )

    with st.expander("Retrieval tuning", expanded=False):
        top_k = st.slider(
            "Retrieved chunks (Top-K)",
            min_value=1,
            max_value=10,
            value=4,
            key="top_k_setting",
        )

        min_score = st.slider(
            "Minimum semantic similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
            key="min_score_setting",
            help=(
                "Filters semantic-search candidates. Hybrid search can also "
                "include strong keyword matches below this cosine-similarity value."
            ),
        )

        st.caption(
            "Chunking settings apply to newly processed documents. "
            "Search settings apply to each question."
        )

    st.divider()
    st.subheader("Storage")

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Documents", store.document_count())
    metric_col2.metric("Vectors", store.count())

    with st.expander("Storage details"):
        st.caption(f"Location: `{STORE_DIR}`")
        st.write(st.session_state.storage_message)

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
        disabled=not st.session_state.chat_history,
    ):
        st.session_state.chat_history = []
        st.rerun()

header_left, header_right = st.columns([5, 1])

with header_left:
    st.title("DocuMind")
    st.caption(
        "Your AI workspace for chatting with PDFs, creating study notes, "
        "generating quizzes, and comparing documents."
    )
    if SETTINGS.demo_mode:
        st.caption("Public demo · read-only sample document library")

with header_right:
    if store.count() > 0:
        st.success("Ready")
    else:
        st.info("No documents")

if store.count() > 0:
    documents = store.get_documents()
    total_pages = sum(
        document.get("page_count", 0)
        for document in documents
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", store.document_count())
    col2.metric("Pages", total_pages)
    col3.metric("Indexed chunks", store.count())

    st.caption(
        f"{len(selected_document_ids)} document(s) selected for search."
    )

    st.divider()

if store.count() > 0:
    st.markdown("### Workspace")
    active_workspace = st.segmented_control(
        "Choose a workspace",
        options=["💬 Chat", "📑 Summarize", "🧠 Quiz", "🗂️ Compare"],
        default="💬 Chat",
        key="workspace_navigation",
        label_visibility="collapsed",
        help="Switch between document chat, summaries, quizzes, and comparisons.",
    )
else:
    active_workspace = None

if active_workspace == "📑 Summarize":
    if store.document_count() > 0:
        st.subheader("📑 AI Document Summarization")
        st.caption(
            "Generate a structured summary from a saved PDF. "
            "Summary generation uses Gemini and includes source page references."
        )

        summary_documents = store.get_documents()
        summary_document_ids = [
            document["document_id"] for document in summary_documents
        ]
        summary_document_names = {
            document["document_id"]: document["filename"]
            for document in summary_documents
        }

        if (
            st.session_state.get("summary_document") is not None
            and st.session_state.summary_document not in summary_document_ids
        ):
            del st.session_state["summary_document"]

        summary_col1, summary_col2 = st.columns([2, 1])

        with summary_col1:
            selected_summary_document = st.selectbox(
                "Select a document to summarize",
                options=summary_document_ids,
                format_func=lambda document_id: summary_document_names[
                    document_id
                ],
                key="summary_document",
            )

        with summary_col2:
            summary_type = st.selectbox(
                "Summary format",
                options=[
                    "Quick Summary",
                    "Detailed Study Notes",
                    "Executive Summary",
                ],
                index=1,
                key="summary_type",
            )

        generate_summary_clicked = st.button(
            "✨ Generate Summary",
            type="primary",
            key="generate_document_summary",
        )

        if generate_summary_clicked:
            selected_chunks = [
                chunk
                for chunk in store.chunks
                if chunk.get("document_id") == selected_summary_document
            ]

            try:
                with st.spinner(
                    "Gemini is analysing the document excerpts..."
                ):
                    summary, summary_sources, coverage = summarize_document(
                        chunks=selected_chunks,
                        filename=summary_document_names[
                            selected_summary_document
                        ],
                        summary_type=summary_type,
                    )

                st.session_state.summary_result = {
                    "document_id": selected_summary_document,
                    "filename": summary_document_names[
                        selected_summary_document
                    ],
                    "summary_type": summary_type,
                    "summary": summary,
                    "sources": summary_sources,
                    "coverage": coverage,
                }

                st.success("Document summary generated successfully.")

            except Exception as error:
                st.session_state.summary_result = None
                st.error(f"Unable to generate summary: {error}")

        summary_result = st.session_state.get("summary_result")

        if (
            summary_result is not None
            and summary_result.get("document_id") in summary_document_ids
        ):

            if summary_result.get("document_id") == selected_summary_document:
                st.markdown("---")
                st.markdown(
                    f"#### Summary: {summary_result['filename']}"
                )
                st.caption(
                    f"Format: {summary_result['summary_type']}"
                )

                coverage = summary_result.get("coverage", {})
                total_chunks = coverage.get("total_chunks", 0)
                used_chunks = coverage.get("used_chunks", 0)
                sampled = coverage.get("sampled", False)

                st.caption(
                    f"Source coverage: {used_chunks} of {total_chunks} chunks used."
                )

                if sampled:
                    st.info(
                        "This is a representative summary based on excerpts "
                        "distributed across the document, not an exhaustive "
                        "summary of every chunk. Review the cited pages for detail."
                    )

                st.markdown(summary_result["summary"])

                summary_sources = summary_result.get("sources", [])
                if summary_sources:
                    with st.expander(
                        f"📄 View summary sources ({len(summary_sources)})"
                    ):
                        for source in summary_sources:
                            st.markdown(
                                f"**[{source['id']}] {source['source']} — "
                                f"Page {source['page']}**"
                            )
                            st.write(source.get("text", ""))
                            st.divider()
                else:
                    st.warning(
                        "The generated summary did not contain validated source references."
                    )

                st.download_button(
                    label="⬇️ Download summary (.md)",
                    data=summary_result["summary"],
                    file_name="DocuMind_Summary.md",
                    mime="text/markdown",
                    key="download_document_summary",
                )

if active_workspace == "🧠 Quiz":
    if store.document_count() > 0:
        st.divider()
        st.subheader("🧠 AI MCQ Generator")
        st.caption(
            "Generate a source-grounded multiple-choice quiz from a saved PDF, "
            "attempt it, and review your score and explanations."
        )

        mcq_documents = store.get_documents()
        mcq_document_ids = [
            document["document_id"] for document in mcq_documents
        ]
        mcq_document_names = {
            document["document_id"]: document["filename"]
            for document in mcq_documents
        }

        if (
            st.session_state.get("mcq_document") is not None
            and st.session_state.mcq_document not in mcq_document_ids
        ):
            del st.session_state["mcq_document"]

        mcq_col1, mcq_col2, mcq_col3 = st.columns([2, 1, 1])

        with mcq_col1:
            selected_mcq_document = st.selectbox(
                "Select a document for the quiz",
                options=mcq_document_ids,
                format_func=lambda document_id: mcq_document_names[document_id],
                key="mcq_document",
            )

        with mcq_col2:
            mcq_difficulty = st.selectbox(
                "Difficulty",
                options=["Easy", "Medium", "Hard", "Mixed"],
                index=1,
                key="mcq_difficulty",
            )

        with mcq_col3:
            mcq_question_count = st.number_input(
                "Number of questions",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="mcq_question_count",
            )

        generate_mcq_clicked = st.button(
            "✨ Generate MCQ Quiz",
            type="primary",
            key="generate_mcq_quiz",
        )

        if generate_mcq_clicked:
            selected_mcq_chunks = [
                chunk
                for chunk in store.chunks
                if chunk.get("document_id") == selected_mcq_document
            ]

            try:
                with st.spinner(
                    "Gemini is creating questions from the document excerpts..."
                ):
                    generated_questions, mcq_coverage = generate_mcqs(
                        chunks=selected_mcq_chunks,
                        filename=mcq_document_names[selected_mcq_document],
                        question_count=int(mcq_question_count),
                        difficulty=mcq_difficulty,
                    )

                quiz_id = uuid.uuid4().hex[:12]
                st.session_state.mcq_result = {
                    "quiz_id": quiz_id,
                    "document_id": selected_mcq_document,
                    "filename": mcq_document_names[selected_mcq_document],
                    "difficulty": mcq_difficulty,
                    "requested_count": int(mcq_question_count),
                    "questions": generated_questions,
                    "coverage": mcq_coverage,
                }
                st.session_state.mcq_attempt = None
                st.success(
                    f"Generated {len(generated_questions)} valid, source-cited question(s)."
                )

            except Exception as error:
                st.session_state.mcq_result = None
                st.session_state.mcq_attempt = None
                st.error(f"Unable to generate MCQs: {error}")

        mcq_result = st.session_state.get("mcq_result")

        if (
            mcq_result is not None
            and mcq_result.get("document_id") in mcq_document_ids
            and mcq_result.get("document_id") == selected_mcq_document
        ):
            coverage = mcq_result.get("coverage", {})
            st.markdown("---")
            st.markdown(f"#### Quiz: {mcq_result['filename']}")
            st.caption(
                f"Difficulty: {mcq_result['difficulty']} · "
                f"Questions: {len(mcq_result['questions'])} · "
                f"Source chunks used: {coverage.get('used_chunks', 0)} "
                f"of {coverage.get('total_chunks', 0)}"
            )

            if coverage.get("sampled"):
                st.info(
                    "Questions were generated from representative excerpts distributed "
                    "across the document, not every chunk. Review the cited pages "
                    "when you need exhaustive coverage."
                )

            quiz_id = mcq_result["quiz_id"]
            questions = mcq_result["questions"]
            submitted_quiz = False

            with st.form(f"mcq_quiz_form_{quiz_id}"):
                for question_index, question_data in enumerate(questions, start=1):
                    st.markdown(
                        f"**Q{question_index}. {question_data['question']}**"
                    )
                    st.radio(
                        "Choose one answer",
                        options=list(("A", "B", "C", "D")),
                        format_func=lambda option, options=question_data["options"]: (
                            f"{option}. {options[option]}"
                        ),
                        index=None,
                        key=f"mcq_choice_{quiz_id}_{question_index}",
                        label_visibility="collapsed",
                    )
                    if question_index < len(questions):
                        st.divider()

                submitted_quiz = st.form_submit_button(
                    "Submit answers",
                    type="primary",
                    use_container_width=True,
                )

            if submitted_quiz:
                answers = {
                    str(index): st.session_state.get(
                        f"mcq_choice_{quiz_id}_{index}"
                    )
                    for index in range(1, len(questions) + 1)
                }
                answered_count = sum(
                    answer is not None for answer in answers.values()
                )
                score = sum(
                    answers[str(index)] == question_data["correct_answer"]
                    for index, question_data in enumerate(questions, start=1)
                )
                st.session_state.mcq_attempt = {
                    "quiz_id": quiz_id,
                    "answers": answers,
                    "score": score,
                    "answered_count": answered_count,
                }

            mcq_attempt = st.session_state.get("mcq_attempt")
            if mcq_attempt and mcq_attempt.get("quiz_id") == quiz_id:
                score = mcq_attempt["score"]
                answered_count = mcq_attempt["answered_count"]
                total_questions = len(questions)
                percentage = (score / total_questions) * 100 if total_questions else 0

                score_col1, score_col2, score_col3 = st.columns(3)
                score_col1.metric("Score", f"{score}/{total_questions}")
                score_col2.metric("Percentage", f"{percentage:.1f}%")
                score_col3.metric("Answered", f"{answered_count}/{total_questions}")

                if score == total_questions:
                    st.success("Excellent! You answered every question correctly.")
                elif score >= total_questions / 2:
                    st.success("Good work! Review the explanations to strengthen your understanding.")
                else:
                    st.info("Review the explanations and source passages, then try another quiz.")

                st.markdown("#### Answer review")
                for question_index, question_data in enumerate(questions, start=1):
                    selected_answer = mcq_attempt["answers"].get(str(question_index))
                    correct_answer = question_data["correct_answer"]
                    is_correct = selected_answer == correct_answer

                    with st.expander(
                        f"Q{question_index}: "
                        f"{'Correct' if is_correct else 'Review answer'}"
                    ):
                        if selected_answer is None:
                            st.warning("You did not select an answer.")
                        elif is_correct:
                            st.success(
                                f"Your answer: {selected_answer}. "
                                f"{question_data['options'][selected_answer]}"
                            )
                        else:
                            st.error(
                                f"Your answer: {selected_answer}. "
                                f"{question_data['options'][selected_answer]}"
                            )

                        st.markdown(
                            f"**Correct answer: {correct_answer}. "
                            f"{question_data['options'][correct_answer]}**"
                        )
                        st.write(question_data["explanation"])

                        question_sources = question_data.get("sources", [])
                        if question_sources:
                            st.markdown("**Supporting document passages**")
                            for source in question_sources:
                                st.caption(
                                    f"[{source['id']}] {source['source']} — "
                                    f"Page {source['page']}"
                                )
                                st.write(source.get("text", ""))

                answer_key_lines = [
                    f"# DocuMind MCQ Quiz — {mcq_result['filename']}",
                    f"Difficulty: {mcq_result['difficulty']}",
                    f"Score: {score}/{total_questions} ({percentage:.1f}%)",
                    "",
                ]
                for question_index, question_data in enumerate(questions, start=1):
                    correct_answer = question_data["correct_answer"]
                    answer_key_lines.extend(
                        [
                            f"## Q{question_index}. {question_data['question']}",
                            *[
                                f"- {option}. {question_data['options'][option]}"
                                for option in ("A", "B", "C", "D")
                            ],
                            f"**Correct answer:** {correct_answer}",
                            f"**Explanation:** {question_data['explanation']}",
                            "**Sources:** " + ", ".join(
                                f"{source['source']} page {source['page']}"
                                for source in question_data.get("sources", [])
                            ),
                            "",
                        ]
                    )

                st.download_button(
                    "⬇️ Download quiz and answer key (.md)",
                    data="\n".join(answer_key_lines),
                    file_name="DocuMind_MCQ_Quiz.md",
                    mime="text/markdown",
                    key=f"download_mcq_{quiz_id}",
                )

if active_workspace == "🗂️ Compare":
    if store.document_count() >= 2:
        st.divider()
        st.subheader("🗂️ Multi-document Intelligence")
        st.caption(
            "Compare two or more PDFs using source-grounded analysis. "
            "DocuMind looks for shared concepts, differences, document-specific "
            "insights, and areas where the available evidence is incomplete."
        )

        comparison_documents = store.get_documents()
        comparison_document_names = {
            document["document_id"]: document["filename"]
            for document in comparison_documents
        }
        comparison_document_options = list(comparison_document_names)

        st.session_state.comparison_document_ids = [
            document_id
            for document_id in st.session_state.comparison_document_ids
            if document_id in comparison_document_names
        ]

        comparison_selected_ids = st.multiselect(
            "Select 2–5 documents to compare",
            options=comparison_document_options,
            format_func=lambda document_id: comparison_document_names[document_id],
            key="comparison_document_ids",
            help="Choose at least two documents. A maximum of five documents can be compared in one request.",
        )

        comparison_focus = st.text_area(
            "What should DocuMind focus on?",
            value=(
                "Compare the main topics, important concepts, definitions, "
                "methods, formulas, assumptions, and conclusions. Identify "
                "similarities and differences, and point out information that "
                "is present in one document but not another."
            ),
            height=100,
            key="comparison_focus",
            help="You can narrow the comparison, for example: compare the algorithms, formulas, assumptions, or conclusions.",
        )

        compare_clicked = st.button(
            "🔍 Compare selected documents",
            type="primary",
            key="compare_selected_documents",
            disabled=len(comparison_selected_ids) < 2 or len(comparison_selected_ids) > 5,
        )

        if len(comparison_selected_ids) == 1:
            st.info("Select at least one more document to compare.")
        elif len(comparison_selected_ids) > 5:
            st.warning("Please select no more than five documents per comparison.")

        if compare_clicked:
            if not comparison_focus.strip():
                st.warning("Enter a comparison focus before generating the report.")
            else:
                comparison_input = []
                for document_id in comparison_selected_ids:
                    document_chunks = [
                        chunk
                        for chunk in store.chunks
                        if chunk.get("document_id") == document_id
                    ]
                    comparison_input.append(
                        {
                            "document_id": document_id,
                            "filename": comparison_document_names[document_id],
                            "chunks": document_chunks,
                        }
                    )

                try:
                    with st.spinner(
                        "Gemini is comparing the selected documents and checking their evidence..."
                    ):
                        report, comparison_sources, comparison_coverage = compare_documents(
                            documents=comparison_input,
                            comparison_focus=comparison_focus,
                        )

                    st.session_state.comparison_result = {
                        "document_ids": list(comparison_selected_ids),
                        "document_names": [
                            comparison_document_names[document_id]
                            for document_id in comparison_selected_ids
                        ],
                        "comparison_focus": comparison_focus,
                        "report": report,
                        "sources": comparison_sources,
                        "coverage": comparison_coverage,
                    }
                    st.success("Multi-document comparison generated successfully.")

                except Exception as error:
                    st.session_state.comparison_result = None
                    st.error(f"Unable to compare documents: {error}")

        comparison_result = st.session_state.get("comparison_result")
        if comparison_result is not None:
            result_ids = comparison_result.get("document_ids", [])
            current_ids = list(comparison_selected_ids)
            same_selection = (
                len(result_ids) == len(current_ids)
                and set(result_ids) == set(current_ids)
            )
            all_sources_still_exist = all(
                document_id in comparison_document_names
                for document_id in result_ids
            )

            if same_selection and all_sources_still_exist:
                st.markdown("---")
                st.markdown("#### Comparison report")
                st.caption(
                    "Compared: " + " · ".join(comparison_result["document_names"])
                )

                coverage_items = comparison_result.get("coverage", {}).get(
                    "documents", []
                )
                if coverage_items:
                    st.markdown("**Evidence coverage**")
                    for item in coverage_items:
                        st.caption(
                            f"{item['filename']}: "
                            f"{item['used_chunks']} of {item['total_chunks']} chunks used"
                        )
                        if item.get("sampled"):
                            st.caption(
                                "Representative excerpts were sampled across this document."
                            )

                st.markdown(comparison_result["report"])

                comparison_sources = comparison_result.get("sources", [])
                if comparison_sources:
                    with st.expander(
                        f"📄 View comparison sources ({len(comparison_sources)})"
                    ):
                        for source in comparison_sources:
                            st.markdown(
                                f"**[{source['id']}] {source['source']} — "
                                f"Page {source['page']}**"
                            )
                            st.write(source.get("text", ""))
                            st.divider()

                download_lines = [
                    "# DocuMind — Multi-document Comparison",
                    "",
                    "**Documents:** " + ", ".join(
                        comparison_result["document_names"]
                    ),
                    "",
                    "**Comparison focus:** " + comparison_result["comparison_focus"],
                    "",
                    comparison_result["report"],
                    "",
                    "## Verified source register",
                ]
                for source in comparison_sources:
                    download_lines.append(
                        f"- [{source['id']}] {source['source']} — page {source['page']}"
                    )

                st.download_button(
                    "⬇️ Download comparison report (.md)",
                    data="\n".join(download_lines),
                    file_name="DocuMind_Multi_Document_Comparison.md",
                    mime="text/markdown",
                    key="download_multi_document_comparison",
                )

    elif store.document_count() == 1:
        st.divider()
        st.subheader("🗂️ Multi-document Intelligence")
        st.info("Upload at least two PDFs to compare documents.")

if store.count() == 0:
    st.markdown(
        """
        <div class="documind-hero">
            <h2>Welcome to your document workspace</h2>
            <p>
                Upload PDFs from the sidebar to build your knowledge
                base. DocuMind will index the content so you can ask
                questions and inspect the supporting passages.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Upload one or more PDFs using the sidebar to get started."
    )

    empty_col1, empty_col2, empty_col3 = st.columns(3)
    with empty_col1:
        st.markdown("**📄 Upload PDFs**")
        st.caption("Build a searchable document library.")
    with empty_col2:
        st.markdown("**🔎 Search content**")
        st.caption("Retrieve relevant passages from your documents.")
    with empty_col3:
        st.markdown("**💬 Ask questions**")
        st.caption("Get answers with document and page references.")

if active_workspace == "💬 Chat":
    if store.count() > 0:
        st.subheader("Conversation")

        if not st.session_state.chat_history:
            with st.chat_message("assistant", avatar="📚"):
                st.markdown(
                    "Hello! I'm DocuMind. Ask a question about the "
                    "documents selected in the sidebar. I'll retrieve "
                    "relevant passages and use them to prepare an answer."
                )

        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(chat["question"])

            with st.chat_message("assistant", avatar="📚"):
                st.markdown(chat["answer"])

                sources = chat.get("sources", [])
                if sources:
                    with st.expander(
                        f"📄 View sources ({len(sources)})"
                    ):
                        for source in sources:
                            st.markdown(
                                f"**[{source['id']}] "
                                f"{source['source']} — "
                                f"Page {source['page']}**"
                            )

                            score = source.get("score")
                            if score is not None:
                                st.caption(
                                    f"Cosine similarity: {score:.4f}"
                                )

                            st.write(source.get("text", ""))
                            st.divider()
                else:
                    st.caption(
                        "No validated source references were returned."
                    )

    if store.count() > 0:
        if not selected_document_ids:
            st.warning(
                "Select at least one document in the sidebar "
                "to enable questions."
            )
        else:
            prompt = st.chat_input(
                "Ask a question about your documents..."
            )

            if prompt:

                st.session_state.chat_history.append(
                    {
                        "question": prompt,
                        "answer": "",
                        "sources": [],
                    }
                )

                try:
                    with st.spinner("Searching your documents..."):
                        query_embedding = embed_query(prompt)

                        retrieved_chunks = store.search_hybrid(
                            query_embedding,
                            query_text=prompt,
                            top_k=int(top_k),
                            min_score=float(min_score),
                            document_ids=set(selected_document_ids),
                        )

                    if not retrieved_chunks:
                        answer = (
                            "I couldn't find sufficiently relevant "
                            "information in the selected documents. "
                            "Try rephrasing your question, selecting "
                            "more documents, or adjusting the similarity "
                            "threshold."
                        )
                        sources = []

                    else:
                        with st.spinner(
                            "Gemini is preparing your answer..."
                        ):
                            answer, sources = generate_answer(
                                prompt,
                                retrieved_chunks,
                            )

                    st.session_state.chat_history[-1]["answer"] = answer
                    st.session_state.chat_history[-1]["sources"] = sources

                except Exception as error:
                    st.session_state.chat_history[-1]["answer"] = (
                        f"Unable to answer your question: {error}"
                    )
                    st.session_state.chat_history[-1]["sources"] = []

                st.rerun()

st.divider()
st.caption(
    "DocuMind · AI document workspace · Streamlit · FAISS · Google Gemini"
)
