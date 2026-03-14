"""View all users in the database."""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config


def main():
    """Display all users."""
    config = load_config()
    db_path = config.storage.sqlite_path
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("\n" + "=" * 80)
    print("📋 רשימת משתמשים במערכת")
    print("=" * 80)
    
    cursor.execute("SELECT username, email, role, created_at, last_login FROM users ORDER BY created_at")
    users = cursor.fetchall()
    
    if not users:
        print("\n❌ אין משתמשים במערכת עדיין.\n")
        conn.close()
        return
    
    print(f"\nסה\"כ {len(users)} משתמשים:\n")
    print(f"{'שם משתמש':<20} {'אימייל':<30} {'תפקיד':<10} {'נוצר ב':<20} {'התחברות אחרונה'}")
    print("-" * 110)
    
    for user in users:
        role_icon = "🔑" if user['role'] == 'admin' else "👤"
        last_login = user['last_login'][:19] if user['last_login'] else "טרם התחבר"
        print(f"{role_icon} {user['username']:<18} {user['email']:<30} {user['role']:<10} {user['created_at'][:19]:<20} {last_login}")
    
    print("\n" + "=" * 80 + "\n")
    
    # Statistics
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    admin_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'user'")
    user_count = cursor.fetchone()[0]
    
    print(f"סטטיסטיקה:")
    print(f"  🔑 Admins: {admin_count}")
    print(f"  👤 Users: {user_count}")
    print()
    
    conn.close()


if __name__ == "__main__":
    main()
