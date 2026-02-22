import React, { useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { resetPassword } from '@/lib/authApi';

export function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');

  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) {
      setError('Invalid reset link. Please click the link in your email again.');
    }
  }, [token]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters long.');
      return;
    }

    setIsLoading(true);

    try {
      await resetPassword(token!, password);
      setSuccess(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset password');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <h1>Set New Password</h1>
          <p>Create a new password for your account</p>
        </div>

        {success ? (
          <div className="auth-success">
            <div className="success-icon">&#x2705;</div>
            <h2>Password Changed!</h2>
            <p>You can now sign in with your new password.</p>
            <Link to="/login" className="auth-button" style={{ display: 'block', textAlign: 'center', textDecoration: 'none', marginTop: '24px' }}>
              Sign In
            </Link>
          </div>
        ) : !token ? (
          <div className="auth-error-box">
            <div className="error-icon">&#x26A0;&#xFE0F;</div>
            <h2>Invalid Link</h2>
            <p>Password reset link is invalid or has expired.</p>
            <Link to="/forgot-password" className="auth-button" style={{ display: 'block', textAlign: 'center', textDecoration: 'none', marginTop: '24px' }}>
              Request New Link
            </Link>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="auth-form">
            {error && <div className="auth-error">{error}</div>}
            <div className="form-group">
              <label htmlFor="password">New Password</label>
              <input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" required disabled={isLoading} minLength={8} />
              <small className="input-hint">At least 8 characters, uppercase, lowercase, and a digit</small>
            </div>
            <div className="form-group">
              <label htmlFor="confirmPassword">Confirm Password</label>
              <input id="confirmPassword" type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="••••••••" required disabled={isLoading} />
            </div>
            <button type="submit" className="auth-button" disabled={isLoading}>
              {isLoading ? 'Saving...' : 'Change Password'}
            </button>
          </form>
        )}

        <div className="auth-footer">
          <p><Link to="/login" className="auth-link">Back to login</Link></p>
        </div>
      </div>

      <style>{`
        .auth-page { min-height: 100vh; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; }
        .auth-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 40px; width: 100%; max-width: 400px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); }
        .auth-header { text-align: center; margin-bottom: 32px; }
        .auth-header h1 { color: #fff; font-size: 24px; font-weight: 600; margin: 0 0 8px 0; }
        .auth-header p { color: rgba(255, 255, 255, 0.6); margin: 0; }
        .auth-form { display: flex; flex-direction: column; gap: 20px; }
        .auth-error { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: #ef4444; padding: 12px 16px; border-radius: 8px; font-size: 14px; }
        .auth-error-box { text-align: center; padding: 20px 0; }
        .auth-error-box .error-icon { font-size: 48px; margin-bottom: 16px; }
        .auth-error-box h2 { color: #f59e0b; font-size: 20px; margin: 0 0 16px 0; }
        .auth-error-box p { color: rgba(255, 255, 255, 0.7); margin: 0; }
        .auth-success { text-align: center; padding: 20px 0; }
        .auth-success .success-icon { font-size: 48px; margin-bottom: 16px; }
        .auth-success h2 { color: #10b981; font-size: 20px; margin: 0 0 16px 0; }
        .auth-success p { color: rgba(255, 255, 255, 0.8); margin: 0; }
        .form-group { display: flex; flex-direction: column; gap: 8px; }
        .form-group label { color: rgba(255, 255, 255, 0.8); font-size: 14px; font-weight: 500; }
        .form-group input { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px; padding: 12px 16px; color: #fff; font-size: 16px; transition: border-color 0.2s, box-shadow 0.2s; }
        .form-group input::placeholder { color: rgba(255, 255, 255, 0.3); }
        .form-group input:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2); }
        .form-group input:disabled { opacity: 0.5; cursor: not-allowed; }
        .input-hint { color: rgba(255, 255, 255, 0.4); font-size: 12px; }
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
