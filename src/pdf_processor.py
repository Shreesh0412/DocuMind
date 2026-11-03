
import pymupdf


def extract_pdf_pages(
    pdf_bytes: bytes,
    filename: str
) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns a list containing:
        - source: PDF filename
        - page: Page number
        - text: Extracted text
    """

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as document:

            # Check whether the PDF is password protected
            if document.needs_pass:
                raise ValueError(
                    f"{filename} is password protected."
                )

            extracted_pages = []

            for page_number, page in enumerate(
                document,
                start=1
            ):

                text = page.get_text(
                    "text",
                    sort=True
                ).strip()

                extracted_pages.append({
                    "source": filename,
                    "page": page_number,
                    "text": text
                })

            return extracted_pages

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            f"Unable to process {filename}: "
            f"{type(error).__name__}"
        ) from error
