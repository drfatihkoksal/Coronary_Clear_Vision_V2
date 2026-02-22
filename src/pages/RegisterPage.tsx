import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '@/lib/AuthContext';

export function RegisterPage() {
  const { register, isLoading } = useAuth();
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [acceptedTerms, setAcceptedTerms] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (!acceptedTerms) {
      setError('You must read and agree to the Terms of Service and Privacy Policy to create an account.');
      return;
    }

    try {
      await register(email, password, name || undefined);
      setSuccess(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed');
    }
  };

  if (success) {
    return (
      <div className="auth-page">
        <div className="auth-card">
          <div className="auth-header">
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>&#x2705;</div>
            <h1 style={{ color: '#10b981' }}>Registration Successful!</h1>
          </div>
          <div style={{ textAlign: 'center' }}>
            <p style={{ color: 'rgba(255, 255, 255, 0.8)', marginBottom: '24px' }}>
              Your account has been created. Please check your email inbox and click the verification link to activate your account.
            </p>
            <p style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '14px', marginBottom: '32px' }}>
              Didn't receive the email? Please check your spam folder.
            </p>
            <Link to="/login" className="auth-button" style={{ display: 'block', textDecoration: 'none', textAlign: 'center' }}>
              Back to Login
            </Link>
          </div>
        </div>
        <style>{`
        .auth-page { min-height: 100vh; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; }
        .auth-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 40px; width: 100%; max-width: 400px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); }
        .auth-header { text-align: center; margin-bottom: 24px; }
        .auth-header h1 { font-size: 24px; font-weight: 600; margin: 0; }
        .auth-button { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: #fff; border: none; border-radius: 8px; padding: 14px 24px; font-size: 16px; font-weight: 600; cursor: pointer; }
        `}</style>
      </div>
    );
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-back">
          <Link to="/" className="back-link">&larr; Back</Link>
        </div>
        <div className="auth-header">
          <h1>Create Account</h1>
          <p>Start analyzing coronary angiography</p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          {error && <div className="auth-error">{error}</div>}

          <div className="form-group">
            <label htmlFor="name">Name (optional)</label>
            <input id="name" type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Dr. John Smith" disabled={isLoading} />
          </div>

          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@hospital.com" required disabled={isLoading} />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" required disabled={isLoading} minLength={8} />
            <span className="password-hint">Min 8 characters, 1 uppercase letter, 1 number</span>
          </div>

          <div className="form-group">
            <label htmlFor="confirmPassword">Confirm Password</label>
            <input id="confirmPassword" type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="••••••••" required disabled={isLoading} />
          </div>

          <button type="submit" className="auth-button" disabled={isLoading}>
            {isLoading ? 'Creating account...' : 'Create Account'}
          </button>

          <div className="terms-checkbox-group">
            <label style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={acceptedTerms}
                onChange={(e) => setAcceptedTerms(e.target.checked)}
                style={{ marginTop: '4px', width: '16px', height: '16px' }}
              />
              <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)', lineHeight: '1.625' }}>
                By ticking this box, I confirm that I have read and agree to the <Link to="/terms" style={{ textDecoration: 'underline', color: 'inherit' }} target="_blank">Terms of Service</Link> and <Link to="/privacy" style={{ textDecoration: 'underline', color: 'inherit' }} target="_blank">Privacy Policy</Link>.
              </span>
            </label>
          </div>
        </form>

        <div className="auth-footer">
          <p>
            Already have an account?{' '}
            <Link to="/login" className="auth-link">Sign in</Link>
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
        .form-group { display: flex; flex-direction: column; gap: 8px; }
        .form-group label { color: rgba(255, 255, 255, 0.8); font-size: 14px; font-weight: 500; }
        .form-group input[type="text"], .form-group input[type="email"], .form-group input[type="password"] { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px; padding: 12px 16px; color: #fff; font-size: 16px; transition: border-color 0.2s, box-shadow 0.2s; }
        .form-group input::placeholder { color: rgba(255, 255, 255, 0.3); }
        .form-group input:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2); }
        .form-group input:disabled { opacity: 0.5; cursor: not-allowed; }
        .password-hint { font-size: 11px; color: rgba(255, 255, 255, 0.5); margin-top: 4px; }
        .auth-button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: #fff; border: none; border-radius: 8px; padding: 14px 24px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }
        .auth-button:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4); }
        .auth-button:disabled { opacity: 0.6; cursor: not-allowed; }
        .terms-checkbox-group { margin-top: -8px; padding: 0 4px; }
        .auth-footer { margin-top: 24px; text-align: center; }
        .auth-footer p { color: rgba(255, 255, 255, 0.6); margin: 0; }
        .auth-link { background: none; border: none; color: #3b82f6; font-size: inherit; cursor: pointer; text-decoration: underline; }
        .auth-link:hover { color: #60a5fa; }
      `}</style>
    </div>
  );
}
