"""
Authentication API Endpoints

User registration, login, token refresh, logout, password reset, email verification.
Adapted from v1 (SQLAlchemy) to v2 (SQLite raw SQL).
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
import secrets
import hashlib
import uuid
import logging

from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, EmailStr

from app.core.auth import (
    UserResponse, TokenPair,
    hash_password, verify_password, validate_password_strength,
    create_token_pair, hash_refresh_token,
)
from app.api.routes.auth_deps import get_current_user
from app.core.email_service import send_password_reset_email, send_verification_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# =============================================================================
# Request/Response Schemas
# =============================================================================

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class AuthResponse(BaseModel):
    user: UserResponse
    tokens: TokenPair


class MessageResponse(BaseModel):
    message: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class VerifyEmailRequest(BaseModel):
    token: str


# =============================================================================
# Helper: get db from request
# =============================================================================

def _get_db(request: Request):
    return request.app.state.db


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def register(request_body: RegisterRequest, request: Request):
    """Register a new user account."""
    db = _get_db(request)
    logger.info(f"Registration attempt for: {request_body.email}")

    # Check if email already exists
    existing = db.connection.execute(
        "SELECT id FROM users WHERE email = ?", (request_body.email,)
    ).fetchone()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Validate password strength
    is_valid, error_msg = validate_password_strength(request_body.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )

    # Create user
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    db.connection.execute(
        "INSERT INTO users (id, email, password_hash, name, is_active, is_verified, created_at) VALUES (?, ?, ?, ?, 1, 0, ?)",
        (user_id, request_body.email, hash_password(request_body.password), request_body.name, now),
    )

    # Create email verification token
    verify_token = secrets.token_urlsafe(32)
    verify_token_hash = hashlib.sha256(verify_token.encode()).hexdigest()
    verify_expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    token_id = str(uuid.uuid4())

    db.connection.execute(
        "INSERT INTO email_verification_tokens (id, user_id, token_hash, expires_at, is_used, created_at) VALUES (?, ?, ?, ?, 0, ?)",
        (token_id, user_id, verify_token_hash, verify_expires_at, now),
    )

    db.connection.commit()

    # Send verification email
    send_verification_email(
        to_email=request_body.email,
        verification_token=verify_token,
        user_name=request_body.name,
    )

    logger.info(f"User registered successfully: {request_body.email}")
    return MessageResponse(
        message="Registration successful! Please check your email and click the verification link to activate your account."
    )


@router.post("/login", response_model=AuthResponse)
async def login(request_body: LoginRequest, request: Request):
    """Authenticate user and return tokens."""
    db = _get_db(request)
    logger.info(f"Login attempt for: {request_body.email}")

    # Find user
    row = db.connection.execute(
        "SELECT id, email, name, password_hash, is_active, is_verified, created_at, last_login_at FROM users WHERE email = ?",
        (request_body.email,),
    ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    user = dict(row)

    # Verify password
    if not verify_password(request_body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Check if account is active
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    # Check if email is verified
    if not user["is_verified"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email address not verified",
        )

    # Update last login
    now = datetime.now(timezone.utc).isoformat()
    db.connection.execute(
        "UPDATE users SET last_login_at = ? WHERE id = ?",
        (now, user["id"]),
    )

    # Create tokens
    token_pair, refresh_hash, refresh_expire = create_token_pair(user["id"])

    # Store refresh token
    token_id = str(uuid.uuid4())
    ip_address = request.client.host if request.client else None
    db.connection.execute(
        "INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at, is_revoked, ip_address, created_at) VALUES (?, ?, ?, ?, 0, ?, ?)",
        (token_id, user["id"], refresh_hash, refresh_expire.isoformat(), ip_address, now),
    )

    db.connection.commit()

    logger.info(f"User logged in: {user['email']}")

    return AuthResponse(
        user=UserResponse(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            is_active=bool(user["is_active"]),
            is_verified=bool(user["is_verified"]),
            created_at=user["created_at"],
        ),
        tokens=token_pair,
    )


@router.post("/refresh", response_model=TokenPair)
async def refresh_token(request_body: RefreshRequest, request: Request):
    """Refresh access token using refresh token (with rotation)."""
    db = _get_db(request)

    # Hash the provided refresh token
    token_hash = hash_refresh_token(request_body.refresh_token)

    # Find the token
    row = db.connection.execute(
        "SELECT id, user_id, expires_at FROM refresh_tokens WHERE token_hash = ? AND is_revoked = 0",
        (token_hash,),
    ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    stored = dict(row)

    # Check expiration
    expires_at = datetime.fromisoformat(stored["expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if datetime.now(timezone.utc) > expires_at:
        db.connection.execute(
            "UPDATE refresh_tokens SET is_revoked = 1 WHERE id = ?", (stored["id"],)
        )
        db.connection.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
        )

    # Get user
    user_row = db.connection.execute(
        "SELECT id, is_active FROM users WHERE id = ?", (stored["user_id"],)
    ).fetchone()

    if user_row is None or not user_row["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Revoke old refresh token (rotation)
    db.connection.execute(
        "UPDATE refresh_tokens SET is_revoked = 1 WHERE id = ?", (stored["id"],)
    )

    # Create new token pair
    token_pair, refresh_hash, refresh_expire = create_token_pair(stored["user_id"])

    # Store new refresh token
    new_token_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    ip_address = request.client.host if request.client else None
    db.connection.execute(
        "INSERT INTO refresh_tokens (id, user_id, token_hash, expires_at, is_revoked, ip_address, created_at) VALUES (?, ?, ?, ?, 0, ?, ?)",
        (new_token_id, stored["user_id"], refresh_hash, refresh_expire.isoformat(), ip_address, now),
    )

    db.connection.commit()
    logger.info(f"Token refreshed for user_id: {stored['user_id']}")

    return token_pair


@router.post("/logout", response_model=MessageResponse)
async def logout(request_body: RefreshRequest, request: Request):
    """Logout by revoking refresh token."""
    db = _get_db(request)
    token_hash = hash_refresh_token(request_body.refresh_token)

    db.connection.execute(
        "UPDATE refresh_tokens SET is_revoked = 1 WHERE token_hash = ?", (token_hash,)
    )
    db.connection.commit()

    return MessageResponse(message="Logged out successfully")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: dict = Depends(get_current_user)):
    """Get current authenticated user's information."""
    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        is_active=bool(user["is_active"]),
        is_verified=bool(user["is_verified"]),
        created_at=user["created_at"],
    )


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(request_body: ForgotPasswordRequest, request: Request):
    """Request password reset email. Always returns success to prevent enumeration."""
    db = _get_db(request)
    logger.info(f"Password reset requested for: {request_body.email}")

    row = db.connection.execute(
        "SELECT id, email, name, is_active FROM users WHERE email = ?",
        (request_body.email,),
    ).fetchone()

    if row is not None and row["is_active"]:
        user = dict(row)

        # Invalidate existing reset tokens
        db.connection.execute(
            "UPDATE password_reset_tokens SET is_used = 1 WHERE user_id = ? AND is_used = 0",
            (user["id"],),
        )

        # Generate new token
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        token_id = str(uuid.uuid4())

        db.connection.execute(
            "INSERT INTO password_reset_tokens (id, user_id, token_hash, expires_at, is_used, created_at) VALUES (?, ?, ?, ?, 0, ?)",
            (token_id, user["id"], token_hash, expires_at, now),
        )
        db.connection.commit()

        send_password_reset_email(
            to_email=user["email"],
            reset_token=token,
            user_name=user["name"],
        )
    else:
        logger.info(f"Password reset requested for non-existent/inactive user: {request_body.email}")

    return MessageResponse(
        message="If an account exists for this email, a password reset link has been sent."
    )


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(request_body: ResetPasswordRequest, request: Request):
    """Reset password using token from email."""
    db = _get_db(request)

    token_hash = hashlib.sha256(request_body.token.encode()).hexdigest()

    row = db.connection.execute(
        "SELECT id, user_id, expires_at FROM password_reset_tokens WHERE token_hash = ? AND is_used = 0",
        (token_hash,),
    ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or already used reset link.",
        )

    stored = dict(row)

    # Check expiration
    expires_at = datetime.fromisoformat(stored["expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if datetime.now(timezone.utc) > expires_at:
        db.connection.execute(
            "UPDATE password_reset_tokens SET is_used = 1 WHERE id = ?", (stored["id"],)
        )
        db.connection.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset link has expired. Please request a new one.",
        )

    # Validate new password
    is_valid, error_msg = validate_password_strength(request_body.new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )

    # Update password
    db.connection.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (hash_password(request_body.new_password), stored["user_id"]),
    )

    # Mark token as used
    db.connection.execute(
        "UPDATE password_reset_tokens SET is_used = 1 WHERE id = ?", (stored["id"],)
    )

    # Revoke all refresh tokens for security
    db.connection.execute(
        "UPDATE refresh_tokens SET is_revoked = 1 WHERE user_id = ? AND is_revoked = 0",
        (stored["user_id"],),
    )

    db.connection.commit()
    logger.info(f"Password reset successful for user_id: {stored['user_id']}")

    return MessageResponse(message="Password changed successfully. You can now sign in with your new password.")


@router.post("/verify-email", response_model=MessageResponse)
async def verify_email(request_body: VerifyEmailRequest, request: Request):
    """Verify user email using token from email."""
    db = _get_db(request)

    token_hash = hashlib.sha256(request_body.token.encode()).hexdigest()

    row = db.connection.execute(
        "SELECT id, user_id, expires_at FROM email_verification_tokens WHERE token_hash = ? AND is_used = 0",
        (token_hash,),
    ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or already used verification link.",
        )

    stored = dict(row)

    # Check expiration
    expires_at = datetime.fromisoformat(stored["expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if datetime.now(timezone.utc) > expires_at:
        db.connection.execute(
            "UPDATE email_verification_tokens SET is_used = 1 WHERE id = ?", (stored["id"],)
        )
        db.connection.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification link has expired. Please request a new one.",
        )

    # Mark user as verified
    db.connection.execute(
        "UPDATE users SET is_verified = 1 WHERE id = ?", (stored["user_id"],)
    )

    # Mark token as used
    db.connection.execute(
        "UPDATE email_verification_tokens SET is_used = 1 WHERE id = ?", (stored["id"],)
    )

    db.connection.commit()
    logger.info(f"Email verified for user_id: {stored['user_id']}")

    return MessageResponse(message="Email verified successfully! You can now sign in.")


@router.post("/resend-verification", response_model=MessageResponse)
async def resend_verification(request_body: ForgotPasswordRequest, request: Request):
    """Resend verification email. Always returns success to prevent enumeration."""
    db = _get_db(request)
    logger.info(f"Verification email resend requested for: {request_body.email}")

    row = db.connection.execute(
        "SELECT id, email, name, is_active, is_verified FROM users WHERE email = ?",
        (request_body.email,),
    ).fetchone()

    if row is not None and row["is_active"] and not row["is_verified"]:
        user = dict(row)

        # Invalidate existing verification tokens
        db.connection.execute(
            "UPDATE email_verification_tokens SET is_used = 1 WHERE user_id = ? AND is_used = 0",
            (user["id"],),
        )

        # Generate new token
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        token_id = str(uuid.uuid4())

        db.connection.execute(
            "INSERT INTO email_verification_tokens (id, user_id, token_hash, expires_at, is_used, created_at) VALUES (?, ?, ?, ?, 0, ?)",
            (token_id, user["id"], token_hash, expires_at, now),
        )
        db.connection.commit()

        send_verification_email(
            to_email=user["email"],
            verification_token=token,
            user_name=user["name"],
        )
    else:
        logger.info(f"Verification resend for non-existent/verified user: {request_body.email}")

    return MessageResponse(
        message="If an unverified account exists for this email, a verification link has been sent."
    )
