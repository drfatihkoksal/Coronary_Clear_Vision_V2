"""
Email Service using Resend

Send transactional emails for password reset, verification, etc.
"""

import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import resend
    RESEND_AVAILABLE = True
except ImportError:
    RESEND_AVAILABLE = False
    logger.warning("Resend package not installed. Email functionality disabled.")


# =============================================================================
# Configuration
# =============================================================================

RESEND_API_KEY = os.getenv("CRA_RESEND_API_KEY") or os.getenv("RESEND_API_KEY")
EMAIL_FROM = os.getenv("CRA_EMAIL_FROM") or os.getenv("EMAIL_FROM", "noreply@rwsanalyser.com")
APP_URL = os.getenv("CRA_APP_URL") or os.getenv("APP_URL", "https://rwsanalyser.com")

if RESEND_API_KEY and RESEND_AVAILABLE:
    resend.api_key = RESEND_API_KEY
    EMAIL_ENABLED = True
    logger.info("Email service initialized with Resend")
else:
    EMAIL_ENABLED = False
    logger.warning("Email service disabled - RESEND_API_KEY not set or resend not installed")


# =============================================================================
# Email Templates
# =============================================================================

def _password_reset_html(reset_url: str, user_name: Optional[str] = None) -> str:
    """Generate password reset email HTML."""
    greeting = f"Merhaba {user_name}," if user_name else "Merhaba,"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sifre Sifirlama</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; padding: 20px 0; }}
            .logo {{ font-size: 24px; font-weight: bold; color: #0ea5e9; }}
            .content {{ background: #f8fafc; border-radius: 8px; padding: 30px; margin: 20px 0; }}
            .button {{ display: inline-block; background: #0ea5e9; color: white !important; text-decoration: none; padding: 12px 30px; border-radius: 6px; font-weight: 600; margin: 20px 0; }}
            .button:hover {{ background: #0284c7; }}
            .footer {{ text-align: center; color: #64748b; font-size: 12px; padding: 20px 0; }}
            .warning {{ color: #64748b; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">Coronary RWS Analyser</div>
            </div>
            <div class="content">
                <p>{greeting}</p>
                <p>Hesabiniz icin sifre sifirlama talebinde bulundunuz. Sifrenizi sifirlamak icin asagidaki butona tiklayin:</p>
                <p style="text-align: center;">
                    <a href="{reset_url}" class="button">Sifremi Sifirla</a>
                </p>
                <p class="warning">Bu link 1 saat icinde gecerlilgini yitirecektir.</p>
                <p class="warning">Eger bu talebi siz yapmadiyseniz, bu emaili gormezden gelebilirsiniz.</p>
            </div>
            <div class="footer">
                <p>Coronary RWS Analyser - Academic Research Tool</p>
                <p>Bu email otomatik olarak gonderilmistir, lutfen yanitlamayin.</p>
            </div>
        </div>
    </body>
    </html>
    """


def _password_reset_text(reset_url: str, user_name: Optional[str] = None) -> str:
    """Generate password reset email plain text."""
    greeting = f"Merhaba {user_name}," if user_name else "Merhaba,"

    return f"""
{greeting}

Hesabiniz icin sifre sifirlama talebinde bulundunuz.

Sifrenizi sifirlamak icin asagidaki linke tiklayin:
{reset_url}

Bu link 1 saat icinde gecerliligini yitirecektir.

Eger bu talebi siz yapmadiyseniz, bu emaili gormezden gelebilirsiniz.

---
Coronary RWS Analyser - Academic Research Tool
    """


# =============================================================================
# Email Functions
# =============================================================================

def send_password_reset_email(
    to_email: str,
    reset_token: str,
    user_name: Optional[str] = None,
) -> bool:
    """Send password reset email."""
    if not EMAIL_ENABLED:
        logger.warning(f"Email disabled - would send password reset to {to_email}")
        return False

    reset_url = f"{APP_URL}/reset-password?token={reset_token}"

    try:
        response = resend.Emails.send({
            "from": f"Coronary RWS Analyser <{EMAIL_FROM}>",
            "to": [to_email],
            "subject": "Sifre Sifirlama - Coronary RWS Analyser",
            "html": _password_reset_html(reset_url, user_name),
            "text": _password_reset_text(reset_url, user_name),
        })

        logger.info(f"Password reset email sent to {to_email}, id: {response.get('id')}")
        return True

    except Exception as e:
        logger.error(f"Failed to send password reset email to {to_email}: {e}")
        return False


def send_welcome_email(
    to_email: str,
    user_name: Optional[str] = None,
) -> bool:
    """Send welcome email to new users."""
    if not EMAIL_ENABLED:
        logger.warning(f"Email disabled - would send welcome email to {to_email}")
        return False

    greeting = f"Merhaba {user_name}," if user_name else "Merhaba,"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; padding: 20px 0; }}
            .logo {{ font-size: 24px; font-weight: bold; color: #0ea5e9; }}
            .content {{ background: #f8fafc; border-radius: 8px; padding: 30px; margin: 20px 0; }}
            .button {{ display: inline-block; background: #0ea5e9; color: white !important; text-decoration: none; padding: 12px 30px; border-radius: 6px; font-weight: 600; }}
            .footer {{ text-align: center; color: #64748b; font-size: 12px; padding: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">Coronary RWS Analyser</div>
            </div>
            <div class="content">
                <p>{greeting}</p>
                <p>Coronary RWS Analyser'a hos geldiniz!</p>
                <p>Artik koroner anjiyografi goruntulerinizi analiz edebilir, RWS hesaplamalari yapabilirsiniz.</p>
                <p style="text-align: center;">
                    <a href="{APP_URL}" class="button">Uygulamaya Git</a>
                </p>
            </div>
            <div class="footer">
                <p>Coronary RWS Analyser - Academic Research Tool</p>
            </div>
        </div>
    </body>
    </html>
    """

    try:
        response = resend.Emails.send({
            "from": f"Coronary RWS Analyser <{EMAIL_FROM}>",
            "to": [to_email],
            "subject": "Hos Geldiniz - Coronary RWS Analyser",
            "html": html_content,
        })

        logger.info(f"Welcome email sent to {to_email}, id: {response.get('id')}")
        return True

    except Exception as e:
        logger.error(f"Failed to send welcome email to {to_email}: {e}")
        return False


def send_verification_email(
    to_email: str,
    verification_token: str,
    user_name: Optional[str] = None,
) -> bool:
    """Send email verification email."""
    if not EMAIL_ENABLED:
        logger.warning(f"Email disabled - would send verification email to {to_email}")
        return False

    verify_url = f"{APP_URL}/verify-email?token={verification_token}"
    greeting = f"Merhaba {user_name}," if user_name else "Merhaba,"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email Dogrulama</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; padding: 20px 0; }}
            .logo {{ font-size: 24px; font-weight: bold; color: #0ea5e9; }}
            .content {{ background: #f8fafc; border-radius: 8px; padding: 30px; margin: 20px 0; }}
            .button {{ display: inline-block; background: #10b981; color: white !important; text-decoration: none; padding: 12px 30px; border-radius: 6px; font-weight: 600; margin: 20px 0; }}
            .button:hover {{ background: #059669; }}
            .footer {{ text-align: center; color: #64748b; font-size: 12px; padding: 20px 0; }}
            .warning {{ color: #64748b; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">Coronary RWS Analyser</div>
            </div>
            <div class="content">
                <p>{greeting}</p>
                <p>Coronary RWS Analyser'a hos geldiniz!</p>
                <p>Hesabinizi aktiflestirmek icin lutfen email adresinizi dogrulayin:</p>
                <p style="text-align: center;">
                    <a href="{verify_url}" class="button">Email Adresimi Dogrula</a>
                </p>
                <p class="warning">Bu link 24 saat icinde gecerliligini yitirecektir.</p>
            </div>
            <div class="footer">
                <p>Coronary RWS Analyser - Academic Research Tool</p>
                <p>Bu email otomatik olarak gonderilmistir, lutfen yanitlamayin.</p>
            </div>
        </div>
    </body>
    </html>
    """

    text_content = f"""
{greeting}

Coronary RWS Analyser'a hos geldiniz!

Hesabinizi aktiflestirmek icin asagidaki linke tiklayin:
{verify_url}

Bu link 24 saat icinde gecerliligini yitirecektir.

---
Coronary RWS Analyser - Academic Research Tool
    """

    try:
        response = resend.Emails.send({
            "from": f"Coronary RWS Analyser <{EMAIL_FROM}>",
            "to": [to_email],
            "subject": "Email Dogrulama - Coronary RWS Analyser",
            "html": html_content,
            "text": text_content,
        })

        logger.info(f"Verification email sent to {to_email}, id: {response.get('id')}")
        return True

    except Exception as e:
        logger.error(f"Failed to send verification email to {to_email}: {e}")
        return False
