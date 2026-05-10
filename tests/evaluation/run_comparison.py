"""
Comparison tool for testing different RAG configurations.
Runs each question through multiple configurations and saves results for manual evaluation.
"""

import json
import logging
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.config import load_config
from src.embeddings.embedder import TextEmbedder
from src.generation.citation_parser import CitationParser
from src.generation.prompt_builder import PromptBuilder
from src.generation.summarizer import Summarizer
from src.retrieval.bm25_store import BM25Store
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigVariant:
    """Represents a specific RAG configuration to test."""

    def __init__(
        self,
        name: str,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        top_k: int = 5,
        initial_candidates: int = 20,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        self.name = name
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.top_k = top_k
        self.initial_candidates = initial_candidates
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            "name": self.name,
            "use_hybrid": self.use_hybrid,
            "use_reranker": self.use_reranker,
            "top_k": self.top_k,
            "initial_candidates": self.initial_candidates,
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
        }


# Define configurations to test
CONFIGURATIONS = [
    ConfigVariant(
        name="vector_only",
        use_hybrid=False,
        use_reranker=False,
        top_k=5,
    ),
    ConfigVariant(
        name="vector_bm25",
        use_hybrid=True,
        use_reranker=False,
        top_k=5,
        bm25_weight=0.3,
        vector_weight=0.7,
    ),
    ConfigVariant(
        name="vector_bm25_reranker",
        use_hybrid=True,
        use_reranker=True,
        top_k=5,
        initial_candidates=20,
        bm25_weight=0.3,
        vector_weight=0.7,
    ),
    ConfigVariant(
        name="vector_only_top10",
        use_hybrid=False,
        use_reranker=False,
        top_k=10,
    ),
    ConfigVariant(
        name="aggressive_bm25",
        use_hybrid=True,
        use_reranker=False,
        top_k=5,
        bm25_weight=0.5,
        vector_weight=0.5,
    ),
]


class ComparisonRunner:
    """Runs comparison tests across multiple configurations."""

    def __init__(
        self,
        config_path: str | None = None,
        retrieval_only: bool = False,
    ):
        """Initialize with base configuration."""
        # Default to config.yaml in the project root (2 levels up from this script)
        if config_path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            config_path = str(project_root / "config.yaml")

        self.config = load_config(config_path)
        self.retrieval_only = retrieval_only

        # Resolve storage paths relative to project root (not CWD)
        project_root = Path(config_path).parent
        if not Path(self.config.storage.chroma_dir).is_absolute():
            self.config.storage.chroma_dir = str(project_root / self.config.storage.chroma_dir)
        if not Path(self.config.storage.bm25_dir).is_absolute():
            self.config.storage.bm25_dir = str(project_root / self.config.storage.bm25_dir)

        self.embedder = TextEmbedder(self.config.embedding)
        self.vector_store = VectorStore(
            persist_directory=self.config.storage.chroma_dir,
        )

        # Initialize vector store
        if not self.vector_store.is_initialized:
            logger.info("Initializing vector store...")
            self.vector_store.initialize()

        # Load BM25 store if available
        bm25_dir = Path(self.config.storage.bm25_dir)
        self.bm25_store = None
        if bm25_dir.exists():
            try:
                self.bm25_store = BM25Store(str(bm25_dir))
                logger.info("BM25 store loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load BM25 store: {e}")

        # Load reranker if available
        self.reranker = None
        if self.config.retrieval.use_reranker:
            try:
                self.reranker = Reranker()
                logger.info("Reranker loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load reranker: {e}")

        # Initialize generation components
        self.prompt_builder = PromptBuilder()
        self.citation_parser = CitationParser()
        self.summarizer = Summarizer(
            config=self.config.generation,
            prompt_builder=self.prompt_builder,
            citation_parser=self.citation_parser,
            anthropic_api_key=self.config.anthropic_api_key,
            openai_api_key=self.config.openai_api_key,
        )

    def run_single_query(
        self, question: str, variant: ConfigVariant
    ) -> dict[str, Any]:
        """Run a single question with a specific configuration variant."""
        logger.info(f"Running '{question}' with config '{variant.name}'")

        # Create retrieval config for this variant
        from src.config import RetrievalConfig

        retrieval_cfg = RetrievalConfig(
            use_hybrid=variant.use_hybrid,
            use_reranker=variant.use_reranker,
            top_k=variant.top_k,
            initial_candidates=variant.initial_candidates,
            bm25_weight=variant.bm25_weight,
            vector_weight=variant.vector_weight,
            min_similarity=self.config.retrieval.min_similarity,
        )

        retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            config=retrieval_cfg,
            bm25_store=self.bm25_store if variant.use_hybrid else None,
            reranker=self.reranker if variant.use_reranker else None,
        )

        # Retrieve sources
        try:
            sources = retriever.search(question, include_context=False)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "config": variant.to_dict(),
                "question": question,
                "sources": [],
                "answer": f"Error during retrieval: {str(e)}",
                "error": str(e),
            }

        # Generate answer
        try:
            if sources:
                if self.retrieval_only:
                    answer = "[Retrieval-only mode] סיכום LLM דולג כדי לאפשר הערכת אחזור ללא תלות בספק API."
                else:
                    generated = self.summarizer.generate(question, sources)
                    if generated:
                        answer = generated.text
                    else:
                        answer = "שגיאה ביצירת תשובה - כל ספקי ה-LLM נכשלו."
            else:
                answer = "לא נמצאו מקורות רלוונטיים לשאלה זו."
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = f"Error during answer generation: {str(e)}"

        # Format sources for output
        sources_data = [
            {
                "book_title": s.chunk.book_title or "Unknown",
                "section_path": s.chunk.section_path,
                "text": s.chunk.text[:500] + "..." if len(s.chunk.text) > 500 else s.chunk.text,
                "score": round(
                    s.rerank_score if s.rerank_score is not None else s.similarity_score,
                    4,
                ),
                "similarity_score": round(s.similarity_score, 4),
                "rerank_score": round(s.rerank_score, 4) if s.rerank_score is not None else None,
            }
            for s in sources[:variant.top_k]
        ]

        return {
            "config": variant.to_dict(),
            "question": question,
            "sources": sources_data,
            "answer": answer,
            "num_sources": len(sources),
        }

    def run_all_questions(
        self, questions_file: str | None = None
    ) -> dict[str, Any]:
        """Run all questions through all configurations."""
        # Default to test_questions.yaml in the same directory as this script
        if questions_file is None:
            script_dir = Path(__file__).parent
            questions_file = str(script_dir / "test_questions.yaml")
        
        # Load questions
        with open(questions_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            questions = data.get("questions", [])

        if not questions:
            raise ValueError(f"No questions found in {questions_file}")

        logger.info(f"Running {len(questions)} questions with {len(CONFIGURATIONS)} configs")

        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(questions),
                "num_configs": len(CONFIGURATIONS),
                "retrieval_only": self.retrieval_only,
                "configurations": [cfg.to_dict() for cfg in CONFIGURATIONS],
            },
            "results": [],
        }

        # Run each question through each configuration
        for q_data in questions:
            question_id = q_data["id"]
            question_text = q_data["question"]

            logger.info(f"\n{'='*60}")
            logger.info(f"Question {question_id}: {question_text}")
            logger.info(f"{'='*60}")

            question_results = {
                "question_id": question_id,
                "question": question_text,
                "variants": [],
            }

            # Test each configuration
            for variant in CONFIGURATIONS:
                result = self.run_single_query(question_text, variant)
                question_results["variants"].append(result)

            results["results"].append(question_results)

        return results

    def save_results(self, results: dict[str, Any], output_dir: str | None = None):
        """Save results to JSON file."""
        # Default to results/ in the same directory as this script
        if output_dir is None:
            script_dir = Path(__file__).parent
            output_dir = str(script_dir / "results")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"comparison_results_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"Results saved to: {filename}")
        logger.info(f"{'='*60}")

        return str(filename)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG configuration comparison for manual evaluation.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config.yaml",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Optional path to questions YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save JSON results",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM answer generation and evaluate retrieval quality only.",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = _parse_args()

    logger.info("Starting RAG Configuration Comparison")
    logger.info(f"Testing {len(CONFIGURATIONS)} configurations")
    if args.retrieval_only:
        logger.info("Running in retrieval-only mode (LLM generation disabled)")

    runner = ComparisonRunner(
        config_path=args.config,
        retrieval_only=args.retrieval_only,
    )
    results = runner.run_all_questions(questions_file=args.questions_file)
    results_file = runner.save_results(results, output_dir=args.output_dir)

    logger.info("\nComparison complete!")
    logger.info(f"Results saved to: {results_file}")
    logger.info("\nNext steps:")
    logger.info("1. Run the evaluation UI to rate the results:")
    logger.info(f"   streamlit run tests/evaluation/evaluate_ui.py -- {results_file}")


if __name__ == "__main__":
    main()
