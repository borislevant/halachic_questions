"""Authentication service - business logic for register, login, verify."""

from pathlib import Path

from src.auth.jwt_handler import create_access_token, decode_access_token
from src.auth.password import hash_password, verify_password
from src.models.user import User, UserCreate, UserLogin
from src.storage.database import (
    create_user,
    get_user_by_username,
    update_last_login,
)


class AuthService:
    """Service for user authentication and authorization.
    
    Attributes:
        db_path: Path to SQLite database.
        jwt_secret: Secret key for JWT signing.
        jwt_algorithm: JWT algorithm (default: HS256).
        jwt_expiry_hours: Token expiry in hours.
    """
    
    def __init__(
        self,
        db_path: str | Path,
        jwt_secret: str,
        jwt_algorithm: str = "HS256",
        jwt_expiry_hours: int = 168,
    ):
        """Initialize AuthService.
        
        Args:
            db_path: Path to SQLite database.
            jwt_secret: Secret key for JWT signing.
            jwt_algorithm: JWT algorithm (default: HS256).
            jwt_expiry_hours: Token expiry in hours (default: 168 = 7 days).
        """
        self.db_path = db_path
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.jwt_expiry_hours = jwt_expiry_hours
    
    def register(self, user_data: UserCreate) -> tuple[User | None, str | None]:
        """Register a new user.
        
        Args:
            user_data: UserCreate schema with username, email, password.
            
        Returns:
            Tuple of (User object, error message).
            If successful, returns (User, None).
            If failed, returns (None, error message).
        """
        # Validate username is lowercase and not taken
        username_lower = user_data.username.lower().strip()
        if len(username_lower) < 3:
            return None, "Username must be at least 3 characters long"
        
        existing_user = get_user_by_username(self.db_path, username_lower)
        if existing_user:
            return None, "Username already taken"
        
        # Validate password strength
        if len(user_data.password) < 8:
            return None, "Password must be at least 8 characters long"
        
        # Hash password and create user
        password_hash = hash_password(user_data.password)
        user = User(
            username=username_lower,
            email=user_data.email.lower().strip(),
            password_hash=password_hash,
            role="user",  # New users are regular users by default
        )
        
        # Save to database
        success = create_user(self.db_path, user)
        if not success:
            return None, "Failed to create user in database"
        
        return user, None
    
    def login(self, credentials: UserLogin) -> tuple[str | None, User | None, str | None]:
        """Authenticate a user and create a JWT token.
        
        Args:
            credentials: UserLogin schema with username and password.
            
        Returns:
            Tuple of (JWT token, User object, error message).
            If successful, returns (token, User, None).
            If failed, returns (None, None, error message).
        """
        username_lower = credentials.username.lower().strip()
        user = get_user_by_username(self.db_path, username_lower)
        
        if not user:
            return None, None, "Invalid username or password"
        
        # Verify password
        if not verify_password(credentials.password, user.password_hash):
            return None, None, "Invalid username or password"
        
        # Update last login timestamp
        update_last_login(self.db_path, user.id)
        
        # Create JWT token
        token = create_access_token(
            user_id=user.id,
            username=user.username,
            role=user.role,
            secret_key=self.jwt_secret,
            algorithm=self.jwt_algorithm,
            expiry_hours=self.jwt_expiry_hours,
        )
        
        return token, user, None
    
    def verify_token(self, token: str) -> tuple[User | None, str | None]:
        """Verify a JWT token and return the associated user.
        
        Args:
            token: JWT token to verify.
            
        Returns:
            Tuple of (User object, error message).
            If valid, returns (User, None).
            If invalid/expired, returns (None, error message).
        """
        payload = decode_access_token(
            token,
            self.jwt_secret,
            self.jwt_algorithm,
        )
        
        if not payload:
            return None, "Invalid or expired token"
        
        # Extract user_id and fetch user from database
        user_id = payload.get("user_id")
        if not user_id:
            return None, "Invalid token payload"
        
        # Import here to avoid circular import
        from src.storage.database import get_user_by_id
        
        user = get_user_by_id(self.db_path, user_id)
        if not user:
            return None, "User not found"
        
        return user, None
