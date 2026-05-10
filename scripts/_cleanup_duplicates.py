"""One-time cleanup: remove duplicate OS book chunks from Chroma, keep one."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.ingestion.pipeline import create_ingestion_pipeline
from src.retrieval.vector_store import VectorStore
from src.storage.database import initialize_database

config = load_config()
initialize_database(config.storage.sqlite_path)

conn = sqlite3.connect(config.storage.sqlite_path)
rows = conn.execute(
    "SELECT id, title, status, chunk_count FROM books WHERE source_path LIKE ?",
    ("%os_book.pdf%",),
).fetchall()
conn.close()

print("DB records for os_book.pdf:")
for r in rows:
    print(f"  id={r[0][:8]}...  title={r[1]}  status={r[2]}  chunks={r[3]}")

vs = VectorStore(persist_directory=config.storage.chroma_dir, collection_name="halachic_texts")
vs.initialize()
print(f"\nTotal items in Chroma before cleanup: {vs.count()}")

for r in rows:
    book_id = r[0]
    result = vs._collection.get(where={"book_id": book_id}, include=["metadatas"])
    print(f"  book_id={book_id[:8]}...  chunks_in_chroma={len(result['ids'])}")

# Delete ALL entries for this source path from DB and Chroma
pipeline = create_ingestion_pipeline(config)
for r in rows:
    book_id = r[0]
    print(f"\nRemoving book_id={book_id[:8]}... from Chroma + DB")
    pipeline.remove_book(book_id)

print(f"\nTotal items in Chroma after cleanup: {vs.count()}")
print("\nNow re-ingesting os_book.pdf cleanly...")
report = pipeline.ingest_book("data/books/os_book.pdf", show_progress=True)
if report.success:
    print(f"✅ {report.book_title}: {report.chunks_created} chunks in {report.processing_time_seconds:.1f}s")
else:
    print(f"❌ Failed: {report.errors}")

print(f"\nTotal items in Chroma after re-ingest: {vs.count()}")
