"""
Authentication Module

JWT-based authentication with refresh tokens.
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from jose import jwt, JWTError

# FIX: Patch bcrypt for passlib compatibility (bcrypt >= 4.0.0 removed __about__)
import bcrypt
if not hasattr(bcrypt, "__about__"):
    class MockAbout:
        __version__ = bcrypt.__version__
    bcrypt.__about__ = MockAbout()

from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Security: In production, JWT_SECRET MUST be set via environment
_jwt_secret_env = os.getenv("CRA_JWT_SECRET") or os.getenv("JWT_SECRET")
_is_production = os.getenv("ENVIRONMENT", "development") == "production"

if _is_production and not _jwt_secret_env:
    raise RuntimeError("JWT_SECRET environment variable is required in production!")

JWT_SECRET = _jwt_secret_env or "development_secret_DO_NOT_USE_IN_PRODUCTION"
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
# Note: truncate_error=False required for bcrypt >= 4.1.0 compatibility with passlib
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__truncate_error=False,
)


# =============================================================================
# Schemas
# =============================================================================

class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # user_id
    exp: datetime
    type: str  # "access" or "refresh"


class TokenPair(BaseModel):
    """Access + refresh token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserCreate(BaseModel):
    """User registration schema."""
    email: EmailStr
    password: str
    name: Optional[str] = None


class UserLogin(BaseModel):
    """User login schema."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response schema (no password)."""
    id: str
    email: str
    name: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: str


# =============================================================================
# Password Utilities
# =============================================================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


# =============================================================================
# Token Utilities
# =============================================================================

def create_access_token(user_id: str) -> Tuple[str, datetime]:
    """Create a JWT access token."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "access",
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, expire


def create_refresh_token() -> Tuple[str, str, datetime]:
    """Create a refresh token. Returns (token, token_hash, expiration)."""
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return token, token_hash, expire


def create_token_pair(user_id: str) -> Tuple[TokenPair, str, datetime]:
    """Create both access and refresh tokens."""
    access_token, access_expire = create_access_token(user_id)
    refresh_token, refresh_hash, refresh_expire = create_refresh_token()

    token_pair = TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=JWT_EXPIRE_MINUTES * 60,
    )

    return token_pair, refresh_hash, refresh_expire


def decode_access_token(token: str) -> Optional[TokenPayload]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        if payload.get("type") != "access":
            logger.warning("Token is not an access token")
            return None

        return TokenPayload(
            sub=payload["sub"],
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            type=payload["type"],
        )
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None


def hash_refresh_token(token: str) -> str:
    """Hash a refresh token for storage."""
    return hashlib.sha256(token.encode()).hexdigest()


# =============================================================================
# Validation
# =============================================================================

def validate_password_strength(password: str) -> Tuple[bool, Optional[str]]:
    """Validate password meets requirements."""
    errors = []

    if len(password) < 8:
        errors.append("at least 8 characters")

    if not any(c.isupper() for c in password):
        errors.append("one uppercase letter (A-Z)")

    if not any(c.islower() for c in password):
        errors.append("one lowercase letter (a-z)")

    if not any(c.isdigit() for c in password):
        errors.append("one number (0-9)")

    if errors:
        if len(errors) == 1:
            return False, f"Password must contain {errors[0]}"
        else:
            requirements = ", ".join(errors[:-1]) + f" and {errors[-1]}"
            return False, f"Password must contain {requirements}"

    return True, None
