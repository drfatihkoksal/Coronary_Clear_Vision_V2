import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { forgotPassword } from '@/lib/authApi';

export function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      await forgotPassword(email);
      setSuccess(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send request');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-back">
          <Link to="/login" className="back-link">&larr; Back to Login</Link>
        </div>
        <div className="auth-header">
          <h1>Forgot Password</h1>
          <p>Enter your email address and we'll send you a link to reset your password</p>
        </div>

        {success ? (
          <div className="auth-success">
            <div className="success-icon">&#9993;&#65039;</div>
            <h2>Email Sent!</h2>
            <p>If an account exists for this email, we have sent a password reset link.</p>
            <p className="success-note">Please check your inbox and spam folder.</p>
            <Link to="/login" className="auth-button" style={{ display: 'block', textAlign: 'center', textDecoration: 'none', marginTop: '24px' }}>
              Back to Login
            </Link>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="auth-form">
            {error && <div className="auth-error">{error}</div>}
            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" required disabled={isLoading} />
            </div>
            <button type="submit" className="auth-button" disabled={isLoading}>
              {isLoading ? 'Sending...' : 'Send Reset Link'}
            </button>
          </form>
        )}

        <div className="auth-footer">
          <p>
            Remember your password?{' '}
            <Link to="/login" className="auth-link">Sign In</Link>
          </p>
        </div>
      </div>

      <style>{`
        .auth-page { min-height: 100vh; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; }
        .auth-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 40px; width: 100%; max-width: 400px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); position: relative; }
        .auth-back { position: absolute; top: 20px; left: 24px; }
        .back-link { color: rgba(255, 255, 255, 0.4); font-size: 13px; text-decoration: none; transition: color 0.2s; }
        .back-link:hover { color: rgba(255, 255, 255, 0.9); }
        .auth-header { text-align: center; margin-bottom: 32px; }
        .auth-header h1 { color: #fff; font-size: 24px; font-weight: 600; margin: 0 0 8px 0; }
        .auth-header p { color: rgba(255, 255, 255, 0.6); margin: 0; }
        .auth-form { display: flex; flex-direction: column; gap: 20px; }
        .auth-error { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: #ef4444; padding: 12px 16px; border-radius: 8px; font-size: 14px; }
        .auth-success { text-align: center; padding: 20px 0; }
        .auth-success .success-icon { font-size: 48px; margin-bottom: 16px; }
        .auth-success h2 { color: #10b981; font-size: 20px; margin: 0 0 16px 0; }
        .auth-success p { color: rgba(255, 255, 255, 0.8); margin: 0 0 8px 0; }
        .auth-success .success-note { color: rgba(255, 255, 255, 0.5); font-size: 14px; }
        .form-group { display: flex; flex-direction: column; gap: 8px; }
        .form-group label { color: rgba(255, 255, 255, 0.8); font-size: 14px; font-weight: 500; }
        .form-group input { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px; padding: 12px 16px; color: #fff; font-size: 16px; transition: border-color 0.2s, box-shadow 0.2s; }
        .form-group input::placeholder { color: rgba(255, 255, 255, 0.3); }
        .form-group input:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2); }
        .form-group input:disabled { opacity: 0.5; cursor: not-allowed; }
        .auth-button { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: #fff; border: none; border-radius: 8px; padding: 14px 24px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }
        .auth-button:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); }
        .auth-button:disabled { opacity: 0.6; cursor: not-allowed; }
        .auth-footer { margin-top: 24px; text-align: center; }
        .auth-footer p { color: rgba(255, 255, 255, 0.6); margin: 0; }
        .auth-link { color: #3b82f6; text-decoration: underline; }
        .auth-link:hover { color: #60a5fa; }
      `}</style>
    </div>
  );
}
