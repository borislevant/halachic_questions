"""Authentication page for login and registration."""

import streamlit as st

from src.auth.auth_service import AuthService
from src.models.user import UserCreate, UserLogin


def render_auth_page(auth_service: AuthService) -> None:
    """Render the authentication page with login and register tabs.
    
    On successful login/register, stores auth_token and user in session_state
    and triggers a rerun to load the main Q&A page.
    
    Args:
        auth_service: Configured AuthService instance.
    """
    st.title("🔐 Halachic Q&A System - Login")
    
    # Disclaimer
    st.markdown(
        '<div class="disclaimer">'
        "⚠️ <strong>Research tool only.</strong> "
        "This application provides AI-generated answers for study purposes. "
        "It is <em>not</em> a substitute for a ruling from a qualified Rabbi."
        "</div>",
        unsafe_allow_html=True,
    )
    
    # Tabs for login and register
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        _render_login_tab(auth_service)
    
    with tab2:
        _render_register_tab(auth_service)


def _render_login_tab(auth_service: AuthService) -> None:
    """Render the login form.
    
    Args:
        auth_service: Configured AuthService instance.
    """
    st.subheader("Login")
    
    with st.form("login_form"):
        username = st.text_input(
            "Username",
            placeholder="Enter username...",
            max_chars=50,
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password...",
            max_chars=100,
        )
        
        submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
        
        if submitted:
            if not username or not password:
                st.error("Please fill in all fields")
                return
            
            credentials = UserLogin(username=username, password=password)
            token, user, error = auth_service.login(credentials)
            
            if error:
                st.error(f"❌ {error}")
            else:
                # Store authentication state
                st.session_state.auth_token = token
                st.session_state.user = user
                st.success(f"✅ Welcome, {user.username}!")
                st.rerun()


def _render_register_tab(auth_service: AuthService) -> None:
    """Render the registration form.
    
    Args:
        auth_service: Configured AuthService instance.
    """
    st.subheader("Register")
    
    with st.form("register_form"):
        username = st.text_input(
            "Username",
            placeholder="Choose username (at least 3 characters)...",
            max_chars=50,
            help="Username will be unique and used for login",
        )
        email = st.text_input(
            "Email Address",
            placeholder="example@email.com",
            max_chars=100,
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="At least 8 characters...",
            max_chars=100,
        )
        password_confirm = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Enter password again...",
            max_chars=100,
        )
        
        submitted = st.form_submit_button("Register", use_container_width=True, type="primary")
        
        if submitted:
            # Validation
            if not username or not email or not password or not password_confirm:
                st.error("Please fill in all fields")
                return
            
            if password != password_confirm:
                st.error("Passwords do not match")
                return
            
            if len(password) < 8:
                st.error("Password must be at least 8 characters long")
                return
            
            # Attempt registration
            user_data = UserCreate(username=username, email=email, password=password)
            user, error = auth_service.register(user_data)
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.success(f"✅ Registration successful! You can now login.")
                st.info("Go to the 'Login' tab to sign in.")
