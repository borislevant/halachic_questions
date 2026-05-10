#!/usr/bin/env python3
"""בדוק כפילויות בתוצאות החדשות"""

import json
from pathlib import Path

# קראו את קובץ התוצאות הכי חדש
results_dir = Path('tests/evaluation/results')
latest_file = max(results_dir.glob('comparison_results_*.json'))

with open(latest_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=== סקירת התוצאות החדשות ===\n')
print(f'קובץ: {latest_file.name}')
print(f'מספר שאלות: {data["metadata"]["num_questions"]}')
print(f'מספר תצורות: {data["metadata"]["num_configs"]}')

print('\n=== בדיקת כפילויות בתשובות (AFTER FIX) ===\n')

# סטטיסטיקה כוללת
total_duplicates = 0
questions_with_dupes = 0

for q_result in data['results']:
    q_id = q_result['question_id']
    q_text = q_result['question'][:50]  # רק 50 תווים ראשונים
    
    has_dupes_in_q = False
    
    for variant in q_result['variants']:
        config_name = variant['config']['name']
        sources = variant['sources']
        num_sources = len(sources)
        
        # בדוק כפילויות בסעיפים
        if num_sources > 0:
            sections = [s['section_path'] for s in sources]
            unique_sections = set(sections)
            
            if len(sections) != len(unique_sections):
                duplicates = len(sections) - len(unique_sections)
                print(f'שאלה {q_id} ({q_text}...)')
                print(f'  {config_name}: {duplicates} כפילויות בסעיפים!')
                print(f'    סעיפים: {sections}')
                total_duplicates += duplicates
                has_dupes_in_q = True
    
    if has_dupes_in_q:
        questions_with_dupes += 1

print(f'\n=== סיכום ===')
print(f'סך הכפילויות שנמצאו: {total_duplicates}')
print(f'מספר שאלות עם כפילויות: {questions_with_dupes} מתוך {len(data["results"])}')

if total_duplicates == 0:
    print('✅ אין כפילויות! ה-fix עבד!')
else:
    print(f'⚠️  עדיין יש {total_duplicates} כפילויות')

# השווה עם התוצאות הקודמות
print(f'\n=== השוואה עם קודם ===')
print('קודם (20260510_113128.json): 84 כפילויות')
if total_duplicates == 0:
    print(f'עכשיו ({latest_file.name}): 0 כפילויות ✅')
else:
    print(f'עכשיו ({latest_file.name}): {total_duplicates} כפילויות ⚠️')
