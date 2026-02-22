import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/lib/AuthContext';
import { resendVerification } from '@/lib/authApi';

export function LoginPage() {
  const navigate = useNavigate();
  const { login, isLoading } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);

  const [isResending, setIsResending] = useState(false);
  const [resendSuccess, setResendSuccess] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResendSuccess(null);

    try {
      await login(email, password);
      navigate('/app');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    }
  };

  const handleResend = async () => {
    if (!email) return;
    setIsResending(true);
    setError(null);
    try {
      await resendVerification(email);
      setResendSuccess('Verification email sent! Please check your inbox and spam folder.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send email');
    } finally {
      setIsResending(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-back">
          <Link to="/" className="back-link">&larr; Back</Link>
        </div>
        <div className="auth-header">
          <h1>Coronary RWS Analyser</h1>
          <p>Sign in to your account</p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          {resendSuccess && (
            <div className="auth-success" style={{ background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', padding: '12px', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
              {resendSuccess}
            </div>
          )}

          {error && (
            <div className="auth-error">
              {error}
              {error === "Email address not verified" && (
                <button
                  type="button"
                  onClick={handleResend}
                  disabled={isResending}
                  style={{
                    display: 'block',
                    marginTop: '8px',
                    background: 'transparent',
                    border: 'none',
                    color: 'inherit',
                    textDecoration: 'underline',
                    cursor: 'pointer',
                    fontSize: '13px',
                    padding: 0
                  }}
                >
                  {isResending ? 'Sending Email...' : 'Resend Verification Email'}
                </button>
              )}
            </div>
          )}

          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              disabled={isLoading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              disabled={isLoading}
            />
          </div>

          <button type="submit" className="auth-button" disabled={isLoading}>
            {isLoading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <div className="auth-footer">
          <p>
            <Link to="/forgot-password" className="auth-link">
              Forgot your password?
            </Link>
          </p>
          <p style={{ marginTop: '12px' }}>
            Don't have an account?{' '}
            <Link to="/register" className="auth-link">
              Create one
            </Link>
          </p>
        </div>
      </div>

      <style>{`
        .auth-page {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
          padding: 20px;
        }
        .auth-card {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 16px;
          padding: 40px;
          width: 100%;
          max-width: 400px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
          position: relative;
        }
        .auth-back { position: absolute; top: 20px; left: 24px; }
        .back-link { color: rgba(255, 255, 255, 0.4); font-size: 13px; text-decoration: none; transition: color 0.2s; }
        .back-link:hover { color: rgba(255, 255, 255, 0.9); }
        .auth-header { text-align: center; margin-bottom: 32px; }
        .auth-header h1 { color: #fff; font-size: 24px; font-weight: 600; margin: 0 0 8px 0; }
        .auth-header p { color: rgba(255, 255, 255, 0.6); margin: 0; }
        .auth-form { display: flex; flex-direction: column; gap: 20px; }
        .auth-error { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: #ef4444; padding: 12px 16px; border-radius: 8px; font-size: 14px; }
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
        .auth-link { background: none; border: none; color: #3b82f6; font-size: inherit; cursor: pointer; text-decoration: underline; }
        .auth-link:hover { color: #60a5fa; }
      `}</style>
    </div>
  );
}
