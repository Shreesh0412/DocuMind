
import streamlit as st

from src.pdf_processor import extract_pdf_pages
from src.text_chunker import chunk_pages
from src.embeddings import embed_texts, embed_query
from src.vector_store import VectorStore
from src.llm import generate_answer


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind")

st.subheader(
    "AI-Powered Document Intelligence System"
)

st.write(
    """
    Upload your PDF documents and ask questions
    about their contents.

    DocuMind retrieves relevant information and
    generates answers with document references.
    """
)

st.divider()


# --------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_stats" not in st.session_state:
    st.session_state.document_stats = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# 3. PDF UPLOAD
# --------------------------------------------------

st.header("Upload Documents")

uploaded_files = st.file_uploader(
    "Choose PDF documents",

    type=["pdf"],

    accept_multiple_files=True
)


# --------------------------------------------------
# 4. DOCUMENT PROCESSING
# --------------------------------------------------

if uploaded_files:

    st.info(
        f"{len(uploaded_files)} document(s) selected."
    )

    if st.button("Process Documents"):

        all_pages = []
        processing_errors = []

        # Step 1: Extract PDF text.
        with st.spinner("Extracting PDF text..."):

            for uploaded_file in uploaded_files:

                try:

                    pages = extract_pdf_pages(
                        uploaded_file.getvalue(),
                        uploaded_file.name
                    )

                    all_pages.extend(pages)

                except ValueError as error:

                    processing_errors.append(
                        str(error)
                    )

        for error in processing_errors:
            st.error(error)

        text_pages = [
            page for page in all_pages
            if page["text"].strip()
        ]

        # Step 2: Generate text chunks.
        all_chunks = chunk_pages(
            text_pages,
            chunk_size=1000,
            overlap=150
        )

        if not all_chunks:

            st.warning(
                "No extractable text was found "
                "in the uploaded PDFs."
            )

        else:

            # Step 3: Generate embeddings.
            with st.spinner(
                "Generating embeddings and building FAISS index..."
            ):

                embeddings = embed_texts(
                    [
                        chunk["text"]
                        for chunk in all_chunks
                    ]
                )

                # Step 4: Build vector index.
                vector_store = VectorStore()

                vector_store.build(
                    all_chunks,
                    embeddings
                )

                # Store in Streamlit session.
                st.session_state.vector_store = vector_store

                st.session_state.document_stats = {
                    "documents": len(uploaded_files),
                    "pages": len(all_pages),
                    "text_pages": len(text_pages),
                    "chunks": len(all_chunks)
                }

                # Reset old conversation on reprocessing.
                st.session_state.chat_history = []

            st.success(
                "Documents processed successfully!"
            )

            st.rerun()


# --------------------------------------------------
# 5. DISPLAY DOCUMENT STATISTICS
# --------------------------------------------------

if st.session_state.vector_store is not None:

    stats = st.session_state.document_stats

    st.divider()

    st.header("Document Processing Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Documents",
        stats["documents"]
    )

    col2.metric(
        "Total Pages",
        stats["pages"]
    )

    col3.metric(
        "Pages with Text",
        stats["text_pages"]
    )

    col4.metric(
        "Total Chunks",
        stats["chunks"]
    )

    st.success(
        "Your documents are indexed and ready for questions."
    )


# --------------------------------------------------
# 6. QUESTION-ANSWERING INTERFACE
# --------------------------------------------------

if st.session_state.vector_store is not None:

    st.divider()

    st.header("Ask Your Documents")

    with st.form("question_form"):

        question = st.text_input(
            "Enter your question",

            placeholder=(
                "Example: Explain virtualization "
                "in cloud computing."
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

                # Step 1: Convert question into embedding.
                with st.spinner(
                    "Understanding your question..."
                ):

                    query_embedding = embed_query(
                        question
                    )

                # Step 2: Retrieve relevant chunks.
                with st.spinner(
                    "Searching your documents..."
                ):

                    retrieved_chunks = (
                        st.session_state.vector_store.search(
                            query_embedding,
                            top_k=4
                        )
                    )

                # Step 3: Generate answer using Gemini.
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


# --------------------------------------------------
# 7. DISPLAY ANSWERS AND SOURCE REFERENCES
# --------------------------------------------------

for index, chat in enumerate(
    st.session_state.chat_history
):

    st.divider()

    st.markdown(
        f"**You:** {chat['question']}"
    )

    st.markdown("### DocuMind's Answer")

    st.markdown(chat["answer"])

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
                f"Cosine similarity: {source['score']:.3f}"
            )

            st.write(
                source["text"]
            )

            st.divider()


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DocuMind | AI-Powered Document Intelligence "
    "System | Built using Python, FAISS and Gemini"
)
