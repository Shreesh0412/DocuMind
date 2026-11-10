# DocuMind UI guide

## Main workspaces

The main page presents one active workspace at a time. Use the segmented
selector below the document metrics to switch between:

1. **Chat** — ask questions against the PDFs selected in the sidebar. Expand
   each answer's source list to inspect the cited passage, page, and retrieval
   score where available.
2. **Summarize** — choose a document and summary format, generate a summary,
   inspect its source register, and download it as Markdown.
3. **Quiz** — choose a document, difficulty, and question count. Generate and
   submit a quiz, then review the score, explanations, and source passages.
4. **Compare** — select two to five documents and enter a comparison focus.
   Generate a source-grounded comparison and download the report.

## Sidebar

- **Add documents:** upload and process PDFs when demo mode is disabled.
- **Document library:** review stored PDFs and remove a document when storage
  recovery is not pending and demo mode is disabled.
- **Search scope:** choose which PDFs Chat should search.
- **Retrieval tuning:** adjust Top-K and minimum semantic similarity only when
  needed; these controls are collapsed by default.
- **Storage:** view document/vector counts and storage status.

## Recommended first-run flow

1. Upload one or more text-based PDFs from the sidebar.
2. Select the relevant PDFs under **Search scope**.
3. Start in **Chat** and ask a specific question.
4. Expand the source list to verify the answer against the cited passage.
5. Use **Summarize**, **Quiz**, or **Compare** for the other document tasks.

In public demo mode, uploads and document removal are disabled. Visitors can
query only the curated sample library configured for the deployment. Demo mode
does not provide per-user privacy or authentication.
