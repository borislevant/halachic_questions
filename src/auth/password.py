"""Password hashing and verification using bcrypt."""

import bcrypt


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt.
    
    Args:
        password: The plaintext password to hash.
        
    Returns:
        The bcrypt hashed password as a string.
    """
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash.
    
    Args:
        password: The plaintext password to verify.
        hashed: The bcrypt hashed password.
        
    Returns:
        True if the password matches, False otherwise.
    """
    password_bytes = password.encode("utf-8")
    hashed_bytes = hashed.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_bytes)
