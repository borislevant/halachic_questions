"""
Quick start script for running evaluation.
"""

import subprocess
import sys
from pathlib import Path


def find_latest_results():
    """Find the most recent results file."""
    results_dir = Path("tests/evaluation/results")
    if not results_dir.exists():
        return None
    
    json_files = list(results_dir.glob("comparison_results_*.json"))
    # Exclude ratings files
    json_files = [f for f in json_files if "_ratings" not in f.name]
    
    if not json_files:
        return None
    
    # Sort by modification time
    latest = max(json_files, key=lambda f: f.stat().st_mtime)
    return str(latest)


def main():
    print("🔍 מערכת הערכת RAG")
    print("=" * 60)
    print()
    print("בחר פעולה:")
    print("1. הרץ השוואת תצורות (run comparison)")
    print("2. פתח ממשק דירוג (evaluate UI)")
    print("3. הצג תוצאות אחרונות (show latest results)")
    print()
    
    choice = input("בחירה (1/2/3): ").strip()
    
    if choice == "1":
        print("\n🚀 מריץ השוואת תצורות...")
        print("זה יכול לקחת כמה דקות תלוי במספר השאלות והתצורות\n")
        subprocess.run([sys.executable, "tests/evaluation/run_comparison.py"])
        
    elif choice == "2":
        latest = find_latest_results()
        if latest:
            print(f"\n📊 פותח ממשק דירוג עם קובץ: {latest}\n")
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                "tests/evaluation/evaluate_ui.py",
                "--",
                latest
            ])
        else:
            print("\n⚠️ לא נמצאו תוצאות. הרץ תחילה השוואת תצורות (אפשרות 1)\n")
            
    elif choice == "3":
        latest = find_latest_results()
        if latest:
            print(f"\n📄 קובץ תוצאות אחרון: {latest}")
            
            # Check if ratings exist
            ratings_file = latest.replace(".json", "_ratings.json")
            if Path(ratings_file).exists():
                print(f"✅ קובץ דירוגים: {ratings_file}")
            else:
                print("⚠️ עדיין אין דירוגים לקובץ זה")
        else:
            print("\n⚠️ לא נמצאו תוצאות\n")
    else:
        print("\n❌ בחירה לא חוקית\n")


if __name__ == "__main__":
    main()
