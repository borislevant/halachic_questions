"""
Interactive UI for manually evaluating and rating RAG comparison results.
Run with: streamlit run tests/evaluation/evaluate_ui.py -- <results_file.json>
"""

import json
import sys
from pathlib import Path

import streamlit as st

# RTL support
st.markdown(
    """
    <style>
    .rtl {
        direction: rtl;
        text-align: right;
    }
    .question-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .variant-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        font-size: 0.9em;
    }
    .answer-text {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-right: 4px solid #0066cc;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_results(filepath: str) -> dict:
    """Load comparison results from JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def save_ratings(results_file: str, ratings: dict):
    """Save ratings to a separate JSON file."""
    ratings_file = results_file.replace(".json", "_ratings.json")
    with open(ratings_file, "w", encoding="utf-8") as f:
        json.dump(ratings, f, ensure_ascii=False, indent=2)
    return ratings_file


def display_sources(sources: list):
    """Display source cards."""
    if not sources:
        st.warning("אין מקורות")
        return

    for i, source in enumerate(sources, 1):
        with st.expander(f"מקור {i}: {source['book_title']} - {source['section_path']} (ציון: {source['score']})"):
            st.markdown(f"<div class='rtl'>{source['text']}</div>", unsafe_allow_html=True)


def display_variant(variant: dict, variant_idx: int, question_id: int):
    """Display a single configuration variant and collect rating."""
    config = variant["config"]
    config_name = config["name"]

    st.markdown(f"### תצורה: {config_name}")

    # Configuration details
    with st.expander("פרטי התצורה"):
        cols = st.columns(3)
        cols[0].metric("Hybrid", "כן" if config["use_hybrid"] else "לא")
        cols[1].metric("Reranker", "כן" if config["use_reranker"] else "לא")
        cols[2].metric("Top K", config["top_k"])
        
        if config["use_hybrid"]:
            st.write(f"BM25 Weight: {config['bm25_weight']}, Vector Weight: {config['vector_weight']}")

    # Sources
    st.write(f"**מספר מקורות שנמצאו:** {variant['num_sources']}")
    display_sources(variant["sources"])

    # Answer
    st.markdown("#### תשובה:")
    st.markdown(
        f"<div class='answer-text rtl'>{variant['answer']}</div>",
        unsafe_allow_html=True,
    )

    # Rating
    st.markdown("---")
    rating_key = f"rating_q{question_id}_v{variant_idx}"
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        rating = st.radio(
            "דרג את התשובה:",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "❌ 1 - גרועה מאוד (טעויות הלכתיות)",
                2: "⚠️ 2 - חלקית/לא מדויקת",
                3: "✅ 3 - סבירה אבל חסרה",
                4: "⭐ 4 - טובה ומדויקת",
                5: "🌟 5 - מצוינת ומקיפה",
            }[x],
            key=rating_key,
            horizontal=False,
        )
    
    with col2:
        st.write("")  # Spacing
        
    # Comments
    comment_key = f"comment_q{question_id}_v{variant_idx}"
    comment = st.text_area(
        "הערות (אופציונלי):",
        key=comment_key,
        height=80,
    )

    return {
        "config_name": config_name,
        "rating": rating,
        "comment": comment,
    }


def main():
    st.set_page_config(
        page_title="RAG Evaluation",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 הערכת תצורות RAG")
    st.markdown("דרג את התשובות מכל תצורה כדי לזהות מה עובד הכי טוב")

    # Load results file
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = st.text_input("נתיב לקובץ תוצאות:", "tests/evaluation/results/comparison_results_latest.json")

    if not results_file or not Path(results_file).exists():
        st.error(f"קובץ לא נמצא: {results_file}")
        st.stop()

    try:
        results = load_results(results_file)
    except Exception as e:
        st.error(f"שגיאה בטעינת הקובץ: {e}")
        st.stop()

    # Display metadata
    metadata = results.get("metadata", {})
    st.sidebar.header("מידע כללי")
    st.sidebar.write(f"**מספר שאלות:** {metadata.get('num_questions', 0)}")
    st.sidebar.write(f"**מספר תצורות:** {metadata.get('num_configs', 0)}")
    st.sidebar.write(f"**תאריך:** {metadata.get('timestamp', 'N/A')}")

    # Configuration selector
    st.sidebar.header("תצורות בבדיקה")
    for cfg in metadata.get("configurations", []):
        st.sidebar.write(f"✅ {cfg['name']}")

    # Question navigation
    all_results = results.get("results", [])
    if not all_results:
        st.warning("אין תוצאות להצגה")
        st.stop()

    question_ids = [r["question_id"] for r in all_results]
    selected_q_id = st.sidebar.selectbox(
        "בחר שאלה:",
        question_ids,
        format_func=lambda x: f"שאלה {x}: {next((r['question'] for r in all_results if r['question_id'] == x), '')[:50]}...",
    )

    # Get selected question
    question_data = next((r for r in all_results if r["question_id"] == selected_q_id), None)
    
    if not question_data:
        st.error("שאלה לא נמצאה")
        st.stop()

    # Display question
    st.markdown(
        f"""
        <div class='question-header rtl'>
            <h2>שאלה {question_data['question_id']}: {question_data['question']}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize ratings storage
    if "all_ratings" not in st.session_state:
        st.session_state.all_ratings = {}

    # Display all variants side by side or stacked
    display_mode = st.radio("תצוגה:", ["מקבילה", "ברצף"], horizontal=True)

    variants = question_data.get("variants", [])
    current_ratings = []

    if display_mode == "מקבילה" and len(variants) <= 3:
        # Side by side for up to 3 variants
        cols = st.columns(len(variants))
        for idx, (col, variant) in enumerate(zip(cols, variants)):
            with col:
                rating_data = display_variant(variant, idx, selected_q_id)
                current_ratings.append(rating_data)
    else:
        # Stacked display
        for idx, variant in enumerate(variants):
            with st.container():
                rating_data = display_variant(variant, idx, selected_q_id)
                current_ratings.append(rating_data)
                st.markdown("---")

    # Save ratings button
    if st.button("💾 שמור דירוגים", type="primary"):
        st.session_state.all_ratings[f"q_{selected_q_id}"] = current_ratings
        ratings_file = save_ratings(results_file, st.session_state.all_ratings)
        st.success(f"✅ דירוגים נשמרו ל-{ratings_file}")

    # Summary statistics (if ratings exist)
    if st.session_state.all_ratings:
        st.sidebar.markdown("---")
        st.sidebar.header("סיכום דירוגים")
        
        # Calculate average rating per configuration
        config_ratings = {}
        for q_ratings in st.session_state.all_ratings.values():
            for variant_rating in q_ratings:
                config_name = variant_rating["config_name"]
                rating = variant_rating["rating"]
                if config_name not in config_ratings:
                    config_ratings[config_name] = []
                config_ratings[config_name].append(rating)
        
        for config_name, ratings in config_ratings.items():
            avg = sum(ratings) / len(ratings)
            st.sidebar.write(f"{config_name}: {avg:.2f} ⭐")


if __name__ == "__main__":
    main()
