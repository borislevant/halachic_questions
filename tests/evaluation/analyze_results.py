"""
Analyze evaluation results and generate summary statistics.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_ratings(ratings_file: str) -> dict[str, Any]:
    """Load ratings from JSON file."""
    with open(ratings_file, encoding="utf-8") as f:
        return json.load(f)


def analyze_ratings(ratings: dict[str, Any]) -> dict[str, Any]:
    """Analyze ratings and compute statistics."""
    # Organize by config
    config_ratings = defaultdict(list)
    config_comments = defaultdict(list)
    
    for question_key, question_ratings in ratings.items():
        for variant in question_ratings:
            config_name = variant["config_name"]
            rating = variant["rating"]
            comment = variant.get("comment", "")
            
            config_ratings[config_name].append(rating)
            if comment:
                config_comments[config_name].append(comment)
    
    # Compute statistics
    stats = {}
    for config_name, rating_list in config_ratings.items():
        n = len(rating_list)
        avg = sum(rating_list) / n if n > 0 else 0
        
        # Count rating distribution
        distribution = {i: rating_list.count(i) for i in range(1, 6)}
        
        stats[config_name] = {
            "count": n,
            "average": round(avg, 2),
            "min": min(rating_list) if rating_list else 0,
            "max": max(rating_list) if rating_list else 0,
            "distribution": distribution,
            "comments": config_comments[config_name],
        }
    
    return stats


def print_summary(stats: dict[str, Any]) -> None:
    """Print formatted summary."""
    print("\n" + "=" * 80)
    print("📊 סיכום תוצאות הערכה")
    print("=" * 80 + "\n")
    
    # Sort by average rating
    sorted_configs = sorted(
        stats.items(),
        key=lambda x: x[1]["average"],
        reverse=True
    )
    
    print("🏆 דירוג תצורות (לפי ממוצע):\n")
    
    for rank, (config_name, data) in enumerate(sorted_configs, 1):
        stars = "⭐" * int(data["average"])
        print(f"{rank}. {config_name}")
        print(f"   ממוצע: {data['average']:.2f} {stars}")
        print(f"   מספר הערכות: {data['count']}")
        print(f"   טווח: {data['min']} - {data['max']}")
        print(f"   התפלגות: ", end="")
        
        # Show distribution
        for rating in range(5, 0, -1):
            count = data["distribution"].get(rating, 0)
            if count > 0:
                print(f"{rating}★×{count} ", end="")
        print("\n")
    
    # Best and worst configs
    print("\n" + "-" * 80)
    best_config = sorted_configs[0]
    worst_config = sorted_configs[-1]
    
    print(f"\n✅ תצורה הטובה ביותר: {best_config[0]} (ממוצע: {best_config[1]['average']})")
    print(f"❌ תצורה הגרועה ביותר: {worst_config[0]} (ממוצע: {worst_config[1]['average']})")
    
    improvement = best_config[1]['average'] - worst_config[1]['average']
    print(f"📈 הפרש: {improvement:.2f} נקודות ({improvement/5*100:.1f}%)")
    
    # Key insights
    print("\n" + "-" * 80)
    print("\n💡 תובנות:\n")
    
    # Check if hybrid helps
    vector_only = stats.get("vector_only", {}).get("average", 0)
    vector_bm25 = stats.get("vector_bm25", {}).get("average", 0)
    
    if vector_only and vector_bm25:
        diff = vector_bm25 - vector_only
        if diff > 0.2:
            print(f"✓ BM25 משפר את התוצאות ב-{diff:.2f} נקודות (+{diff/5*100:.1f}%)")
        elif diff < -0.2:
            print(f"✗ BM25 מחמיר את התוצאות ב-{abs(diff):.2f} נקודות ({diff/5*100:.1f}%)")
        else:
            print(f"≈ BM25 לא משפיע משמעותית ({diff:+.2f} נקודות)")
    
    # Check if reranker helps
    vector_bm25_reranker = stats.get("vector_bm25_reranker", {}).get("average", 0)
    
    if vector_bm25 and vector_bm25_reranker:
        diff = vector_bm25_reranker - vector_bm25
        if diff > 0.2:
            print(f"✓ Reranker משפר את התוצאות ב-{diff:.2f} נקודות (+{diff/5*100:.1f}%)")
        elif diff < -0.2:
            print(f"✗ Reranker מחמיר את התוצאות ב-{abs(diff):.2f} נקודות ({diff/5*100:.1f}%)")
        else:
            print(f"≈ Reranker לא משפיע משמעותית ({diff:+.2f} נקודות)")
    
    # Comments summary
    print("\n" + "-" * 80)
    print("\n📝 הערות שניתנו:\n")
    
    has_comments = False
    for config_name, data in sorted_configs:
        if data["comments"]:
            has_comments = True
            print(f"\n{config_name}:")
            for i, comment in enumerate(data["comments"][:3], 1):  # Show first 3
                print(f"  {i}. {comment}")
            if len(data["comments"]) > 3:
                print(f"  ... ועוד {len(data['comments']) - 3} הערות")
    
    if not has_comments:
        print("אין הערות.")
    
    print("\n" + "=" * 80 + "\n")


def save_summary(stats: dict[str, Any], output_file: str) -> None:
    """Save summary to JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"📁 סיכום נשמר ל-{output_file}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # Try to find latest ratings file
        results_dir = Path("tests/evaluation/results")
        ratings_files = list(results_dir.glob("*_ratings.json"))
        
        if not ratings_files:
            print("❌ לא נמצאו קבצי דירוגים")
            print("\nשימוש: python analyze_results.py <ratings_file.json>")
            sys.exit(1)
        
        ratings_file = max(ratings_files, key=lambda f: f.stat().st_mtime)
        print(f"📂 משתמש בקובץ דירוגים אחרון: {ratings_file}")
    else:
        ratings_file = sys.argv[1]
    
    if not Path(ratings_file).exists():
        print(f"❌ קובץ לא נמצא: {ratings_file}")
        sys.exit(1)
    
    # Load and analyze
    ratings = load_ratings(ratings_file)
    
    if not ratings:
        print("⚠️ הקובץ ריק או לא מכיל דירוגים")
        sys.exit(0)
    
    stats = analyze_ratings(ratings)
    
    # Print summary
    print_summary(stats)
    
    # Save summary
    summary_file = str(ratings_file).replace("_ratings.json", "_summary.json")
    save_summary(stats, summary_file)


if __name__ == "__main__":
    main()
