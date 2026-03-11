"""Permission checks for role-based access control."""

from src.models.book import Book
from src.models.user import User


def can_add_shared_book(user: User) -> bool:
    """Check if user can add books to the shared library.
    
    Only admins can add shared books.
    
    Args:
        user: The user to check.
        
    Returns:
        True if user is admin, False otherwise.
    """
    return user.role == "admin"


def can_delete_book(user: User, book: Book) -> bool:
    """Check if user can delete a book.
    
    Users can delete their own private books.
    Admins can delete any book (shared or private).
    
    Args:
        user: The user attempting deletion.
        book: The book to delete.
        
    Returns:
        True if user has permission, False otherwise.
    """
    if user.role == "admin":
        return True
    
    # Regular users can only delete their own books
    return book.user_id == user.id


def can_delete_query(user_id: str, query_owner_id: str) -> bool:
    """Check if user can delete a query.
    
    Users can only delete their own queries.
    
    Args:
        user_id: The user attempting deletion.
        query_owner_id: The owner of the query.
        
    Returns:
        True if same user, False otherwise.
    """
    return user_id == query_owner_id
