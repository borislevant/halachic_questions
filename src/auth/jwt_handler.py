"""JWT token encoding and decoding for session management."""

from datetime import datetime, timedelta

from jose import JWTError, jwt


def create_access_token(
    user_id: str,
    username: str,
    role: str,
    secret_key: str,
    algorithm: str = "HS256",
    expiry_hours: int = 168,
) -> str:
    """Create a JWT access token for a user.
    
    Args:
        user_id: The user's unique ID.
        username: The user's username.
        role: The user's role ('admin' or 'user').
        secret_key: Secret key for JWT signing.
        algorithm: JWT signing algorithm (default: HS256).
        expiry_hours: Token expiry in hours (default: 168 = 7 days).
        
    Returns:
        Encoded JWT token as a string.
    """
    expiry = datetime.utcnow() + timedelta(hours=expiry_hours)
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "exp": expiry,
        "iat": datetime.utcnow(),
    }
    token = jwt.encode(payload, secret_key, algorithm=algorithm)
    return token


def decode_access_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256",
) -> dict | None:
    """Decode and verify a JWT access token.
    
    Args:
        token: The JWT token to decode.
        secret_key: Secret key used for signing.
        algorithm: JWT algorithm used (default: HS256).
        
    Returns:
        The decoded payload as a dict if valid, None if invalid or expired.
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except JWTError:
        return None
