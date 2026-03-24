"""Cached service initialization for the Streamlit UI.

All heavy objects (embedding model, vector store, pipeline) are created once
per server lifetime via @st.cache_resource and shared across rerenders.
"""

import logging

import streamlit as st

from src.config import AppConfig, load_config
from src.embeddings.embedder import TextEmbedder
from src.generation.citation_parser import CitationParser
from src.generation.prompt_builder import PromptBuilder
from src.generation.summarizer import Summarizer
from src.ingestion.chunker import HalachicChunker
from src.ingestion.parser import BookParser
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.bm25_store import BM25Store
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading models… (first run may take a minute)")
def load_services() -> dict:
    """Load and cache all application services.

    Instantiates the embedding model, vector store, retriever,
    summarizer, and ingestion pipeline. This function executes once per
    Streamlit server process; subsequent calls return the cached objects.

    Returns:
        Dict with keys: ``config``, ``retriever``, ``summarizer``, ``pipeline``.
    """
    logger.info("Initialising application services")
    config: AppConfig = load_config()

    # Shared embedding model — used by both retriever and ingestion pipeline
    embedder = TextEmbedder(config.embedding)

    # Vector store — shared so retriever and pipeline operate on the same data
    vector_store = VectorStore(
        persist_directory=config.storage.chroma_dir,
        collection_name="halachic_texts",
    )
    vector_store.initialize()

    # BM25 store — shared for hybrid retrieval
    bm25_store = None
    if config.retrieval.use_hybrid:
        logger.info("Hybrid retrieval enabled, initializing BM25 store")
        bm25_store = BM25Store(bm25_dir=config.storage.bm25_dir)
        # Try to load existing index
        if bm25_store.load_index():
            logger.info("BM25 index loaded successfully (%d chunks)", bm25_store.chunk_count)
        else:
            logger.warning("No BM25 index found. Will be built during ingestion.")

    # Initialize reranker if enabled
    reranker = None
    if config.retrieval.use_reranker:
        logger.info("Reranker enabled, initializing cross-encoder model")
        reranker = Reranker(device=config.embedding.device)

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        config=config.retrieval,
        bm25_store=bm25_store,
        reranker=reranker,
    )

    summarizer = Summarizer(
        config=config.generation,
        prompt_builder=PromptBuilder(),
        citation_parser=CitationParser(),
        anthropic_api_key=config.anthropic_api_key,
        openai_api_key=config.openai_api_key,
    )

    pipeline = IngestionPipeline(
        config=config,
        parser=BookParser(),
        chunker=HalachicChunker(config.chunking),
        embedder=embedder,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    logger.info("Application services ready")
    return {
        "config": config,
        "retriever": retriever,
        "summarizer": summarizer,
        "pipeline": pipeline,
    }
