"""Create an admin user - run this once during initial setup."""

import sys
from getpass import getpass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auth.password import hash_password
from src.config import load_config
from src.models.user import User
from src.storage.database import create_user, get_user_by_username, initialize_database


def main():
    """Create an admin user interactively."""
    print("=" * 60)
    print("יצירת משתמש אדמין - Halachic Q&A")
    print("=" * 60)
    print()
    
    # Load config
    config = load_config()
    db_path = config.storage.sqlite_path
    
    # Initialize database if needed
    initialize_database(db_path)
    
    # Get user input
    print("הכנס פרטי משתמש אדמין:")
    username = input("שם משתמש (אנגלית): ").strip().lower()
    
    if not username or len(username) < 3:
        print("❌ שם משתמש חייב להיות לפחות 3 תווים")
        return
    
    # Check if username exists
    existing = get_user_by_username(db_path, username)
    if existing:
        print(f"❌ שם המשתמש '{username}' כבר קיים במערכת")
        return
    
    email = input("אימייל: ").strip().lower()
    if not email or "@" not in email:
        print("❌ כתובת אימייל לא תקינה")
        return
    
    password = getpass("סיסמה (לא תוצג על המסך): ")
    if len(password) < 8:
        print("❌ סיסמה חייבת להיות לפחות 8 תווים")
        return
    
    password_confirm = getpass("אימות סיסמה: ")
    if password != password_confirm:
        print("❌ הסיסמאות אינן תואמות")
        return
    
    # Create admin user
    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
        role="admin",
    )
    
    success = create_user(db_path, user)
    if success:
        print()
        print("=" * 60)
        print("✅ משתמש אדמין נוצר בהצלחה!")
        print(f"   שם משתמש: {username}")
        print(f"   תפקיד: admin")
        print("=" * 60)
        print()
        print("כעת תוכל להתחבר למערכת עם הפרטים שהזנת.")
    else:
        print("❌ שגיאה ביצירת המשתמש")


if __name__ == "__main__":
    main()
