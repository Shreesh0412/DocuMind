# DocuMind deployment checklist

## Public demo safety

DocuMind currently uses one shared vector-store directory and does not include
user authentication or per-user document isolation. For a public portfolio demo:

1. Deploy only curated, non-sensitive sample PDFs.
2. Set `DOCUMIND_DEMO_MODE=true` in the hosting provider's environment/secrets.
3. Keep the demo library read-only. Demo mode disables document uploads and
   removal in the UI and also blocks upload processing in the application layer.
4. Set `GEMINI_API_KEY` in the hosting provider's secret manager. Do not commit
   `.env`, API keys, private PDFs, or `data/documind_store/`.
5. Confirm that the deployed instance contains only the sample documents you
   intend every visitor to be able to query.

**Important:** Demo mode is a read-only switch, not authentication or tenant
isolation. All visitors can query the same document library. Do not deploy
private or user-uploaded documents to this shared demo.

## Resource limits

The defaults are deliberately conservative and can be changed through the
hosting provider's environment variables:

| Variable | Default | Meaning |
|---|---:|---|
| `MAX_PDF_FILE_SIZE_MB` | 20 | Maximum size of one PDF |
| `MAX_PDF_PAGES` | 500 | Maximum pages in one PDF |
| `MAX_BATCH_FILES` | 5 | Maximum PDFs in one batch |
| `MAX_BATCH_SIZE_MB` | 50 | Maximum combined upload size |
| `MAX_CHUNKS_PER_BATCH` | 5000 | Maximum chunks embedded in one batch |
| `DOCUMIND_DEMO_MODE` | `false` | Disable uploads and document removal |
| `DOCUMIND_LOG_LEVEL` | `INFO` | Console log verbosity |

Streamlit's `.streamlit/config.toml` also sets `server.maxUploadSize = 20` MB.
If you raise `MAX_PDF_FILE_SIZE_MB`, update this Streamlit setting to the same
value. The application checks per-file and combined batch sizes again before
processing.

## Storage and backups

The local vector store now writes immutable snapshots under
`data/documind_store/snapshots/` and atomically updates the `CURRENT` pointer
only after a snapshot is complete. Up to three recent generations are retained.
If the current snapshot fails integrity checks, loading tries an earlier valid
snapshot. Stores created with the previous flat-file layout remain readable;
the next save migrates them to snapshots.

For a manual local backup, stop the app and run:

```bash
tar -czf DocuMind_store_backup.tar.gz -C data documind_store
```

For production, use a persistent volume or managed object/database storage for
these snapshots. Some hosted application filesystems are ephemeral and may be
reset during rebuilds or restarts. Keep an external backup for any library that
must survive a deployment.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Set secrets in the hosting provider rather than uploading `.env`. Launch with
`streamlit run app.py` from the repository root.

## Reproducible dependency lock

`requirements.txt` constrains supported major-version ranges. Once the project
has passed its tests in the release environment, create and commit a full
transitive lock from that verified environment:

```bash
./scripts/create_lockfile.sh
```

The script compiles the project, runs the offline test suite, checks installed
package consistency with `pip check`, and only then writes
`requirements.lock.txt`. Use `pip install -r requirements.lock.txt` when
recreating that exact verified environment. Regenerate the lock deliberately
when dependencies are upgraded; do not create it from an untested environment.

## Pre-release checks

```bash
python -m compileall -q app.py src scripts tests
python -m unittest discover -s tests -v
```

After deployment, verify that the sample library loads, a supported question
returns cited sources, and uploads/removal are unavailable while demo mode is
on. Do not use private documents for this smoke test.
