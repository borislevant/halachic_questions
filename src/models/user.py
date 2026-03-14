"""User model for authentication and authorization."""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    """User model with authentication details.
    
    Attributes:
        id: Unique identifier (UUID).
        username: Unique username for login (lowercase).
        email: User's email address.
        password_hash: Bcrypt hashed password (never store plaintext).
        role: User role - 'admin' or 'user' (default: 'user').
        created_at: Account creation timestamp.
        last_login: Last successful login timestamp (optional).
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    username: str
    email: EmailStr
    password_hash: str
    role: str = "user"  # 'admin' or 'user'
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: datetime | None = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class UserCreate(BaseModel):
    """Schema for user registration.
    
    Used for registration form - does not include hashed password yet.
    """
    
    username: str
    email: EmailStr
    password: str
    
    
class UserLogin(BaseModel):
    """Schema for user login.
    
    Used for login form validation.
    """
    
    username: str
    password: str
