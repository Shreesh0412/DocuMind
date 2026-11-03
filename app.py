
import streamlit as st

from src.pdf_processor import extract_pdf_pages


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

    # Dashboard metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Documents uploaded",
        len(uploaded_files)
    )

    col2.metric(
        "Total pages",
        len(all_pages)
    )

    col3.metric(
        "Pages with extracted text",
        len(text_pages)
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

else:

    st.info(
        "Upload one or more PDF documents to begin."
    )
