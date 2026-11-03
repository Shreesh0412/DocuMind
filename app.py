
import streamlit as st

from src.pdf_processor import extract_pdf_pages
from src.text_chunker import chunk_pages

# Application configuration
st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide"
)


# Main interface
st.title("📚 DocuMind")

st.subheader(
    "AI-Powered Document Intelligence System"
)

st.write(
    "Upload your PDF documents and prepare "
    "them for intelligent question answering."
)

st.divider()


# PDF uploader
uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    all_pages = []

    for uploaded_file in uploaded_files:

        try:
            pages = extract_pdf_pages(
                uploaded_file.getvalue(),
                uploaded_file.name
            )

            all_pages.extend(pages)

        except ValueError as error:
            st.error(str(error))

    # Separate pages containing selectable text
    text_pages = [
        page for page in all_pages
        if page["text"]
    ]
    # Generate overlapping text chunks
    all_chunks = chunk_pages(
        all_pages,
        chunk_size=1000,
        overlap=150
    )
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Documents uploaded",
        len(uploaded_files)
    )

    col2.metric(
        "Total pages",
        len(all_pages)
    )

    col3.metric(
        "Pages with text",
        len(text_pages)
    )

    col4.metric(
        "Chunks generated",
        len(all_chunks)
    )

    st.divider()

    if not text_pages:

        st.warning(
            "No selectable text was found. "
            "The PDFs may contain scanned images."
        )

    else:

        st.success(
            "PDF text extraction successful!"
        )

        # Select document page
        selected_index = st.selectbox(
            "Select a page to preview",

            options=range(len(text_pages)),

            format_func=lambda index: (
                f"{text_pages[index]['source']} "
                f"- Page {text_pages[index]['page']}"
            )
        )

        selected_page = text_pages[selected_index]

        # Display extracted content
        st.subheader("Extracted Content")

        st.text_area(
            "PDF Text",
            value=selected_page["text"],
            height=350,
            disabled=True
        )

        st.caption(
            f"Source: {selected_page['source']} "
            f"| Page: {selected_page['page']}"
        )

        # --------------------------------
        # Chunk Preview
        # --------------------------------

        st.divider()

        st.subheader("Text Chunk Preview")

        if all_chunks:

            selected_chunk_index = st.selectbox(
                "Select a chunk",

                options=range(len(all_chunks)),

                format_func=lambda index: (
                    f"Chunk {index + 1} - "
                    f"{all_chunks[index]['source']} "
                    f"(Page {all_chunks[index]['page']})"
                )
            )

            selected_chunk = all_chunks[
                selected_chunk_index
            ]

            st.write(
                f"Chunk ID: {selected_chunk['chunk_id']}"
            )

            st.text_area(
                "Chunk Content",
                value=selected_chunk["text"],
                height=250,
                disabled=True
            )

        else:

            st.warning("No text chunks were generated.")


else:

    st.info(
        "Upload one or more PDF documents to begin."
    )
