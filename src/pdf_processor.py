import pymupdf

def extract_pdf_pages(pdf_bytes: bytes, filename: str) -> list[dict]:
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError(f"{filename} is password protected.")

            return [
                {
                    "source": filename,
                    "page": page_number,
                    "text": page.get_text("text", sort=True).strip(),
                }
                for page_number, page in enumerate(document, start=1)
            ]
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            f"Could not process {filename}: {type(error).__name__}"
        ) from error
