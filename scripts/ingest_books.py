"""Ingest one or more book files into the Halachic Q&A indexes.

Usage examples:
    python scripts/ingest_books.py --file "data/books/my_book.pdf"
    python scripts/ingest_books.py --dir "data/books" --recursive
    python scripts/ingest_books.py --file "book.txt" --author "Yosef Karo"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.ingestion.parser import SUPPORTED_FORMATS
from src.ingestion.pipeline import create_ingestion_pipeline
from src.storage.database import initialize_database


def _supported_suffixes() -> tuple[str, ...]:
    """Return supported file extensions for ingestion."""
    return tuple(SUPPORTED_FORMATS.keys())


def _iter_supported_files(base_dir: Path, recursive: bool) -> list[Path]:
    """Collect supported files from a directory."""
    patterns = [f"*{ext}" for ext in _supported_suffixes()]
    files: list[Path] = []

    for pattern in patterns:
        matched = base_dir.rglob(pattern) if recursive else base_dir.glob(pattern)
        files.extend(matched)

    # Remove duplicates (e.g. if patterns overlap) and sort for deterministic order
    return sorted(set(files))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest Halachic books into vector/BM25 stores.",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single book file to ingest",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to a directory of book files to ingest",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when using --dir",
    )
    parser.add_argument(
        "--author",
        type=str,
        default="",
        help="Optional author override for all ingested files",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help=(
            "Optional owner user_id for private books. "
            "If omitted, books are ingested as shared/public."
        ),
    )

    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.error("You must provide either --file or --dir")

    return args


def main() -> None:
    """CLI entry point for book ingestion."""
    args = _parse_args()

    config = load_config()

    # Ensure required storage paths and DB exist
    Path(config.storage.books_dir).mkdir(parents=True, exist_ok=True)
    Path(config.storage.processed_dir).mkdir(parents=True, exist_ok=True)
    Path(config.storage.chroma_dir).mkdir(parents=True, exist_ok=True)
    Path(config.storage.bm25_dir).mkdir(parents=True, exist_ok=True)
    initialize_database(config.storage.sqlite_path)

    pipeline = create_ingestion_pipeline(config)

    files_to_ingest: list[Path] = []

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        files_to_ingest.append(file_path)

    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        dir_files = _iter_supported_files(dir_path, recursive=args.recursive)
        files_to_ingest.extend(dir_files)

    # Deduplicate if both --file and --dir include the same path
    files_to_ingest = sorted(set(files_to_ingest))

    if not files_to_ingest:
        supported = ", ".join(_supported_suffixes())
        print("⚠️ No supported book files found.")
        print(f"Supported extensions: {supported}")
        return

    print("=" * 70)
    print("📚 Book Ingestion")
    print("=" * 70)
    print(f"Files to ingest: {len(files_to_ingest)}")
    print(f"Owner: {'shared/public' if args.user_id is None else args.user_id}")
    print()

    success_count = 0
    failed_count = 0

    for idx, file_path in enumerate(files_to_ingest, start=1):
        print(f"[{idx}/{len(files_to_ingest)}] Ingesting: {file_path}")
        report = pipeline.ingest_book(
            file_path=file_path,
            author=args.author.strip(),
            user_id=args.user_id,
            show_progress=True,
        )

        if report.success:
            success_count += 1
            print(
                "  ✅ Success: "
                f"{report.book_title} "
                f"({report.chunks_created} chunks, {report.processing_time_seconds:.1f}s)"
            )
            if report.warnings:
                for warning in report.warnings:
                    print(f"  ⚠️ Warning: {warning}")
        else:
            failed_count += 1
            error = report.errors[0] if report.errors else "Unknown ingestion error"
            print(f"  ❌ Failed: {error}")

        print()

    print("=" * 70)
    print(
        f"Done. Succeeded: {success_count} | Failed: {failed_count} | "
        f"Total: {len(files_to_ingest)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
