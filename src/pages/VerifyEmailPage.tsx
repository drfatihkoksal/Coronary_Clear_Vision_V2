import { useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { verifyEmail } from '@/lib/authApi';

export function VerifyEmailPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');

  const [isLoading, setIsLoading] = useState(true);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) {
      setError('Invalid verification link.');
      setIsLoading(false);
      return;
    }

    verifyEmail(token)
      .then(() => {
        setSuccess(true);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Verification failed');
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [token]);

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <h1>Email Verification</h1>
        </div>

        {isLoading ? (
          <div className="auth-loading">
            <div className="loading-spinner"></div>
            <p>Verifying your email address...</p>
          </div>
        ) : success ? (
          <div className="auth-success">
            <div className="success-icon">&#x2705;</div>
            <h2>Email Verified!</h2>
            <p>Your account has been activated. You can now sign in to the application.</p>
            <Link to="/login" className="auth-button" style={{ display: 'block', textAlign: 'center', textDecoration: 'none', marginTop: '24px' }}>
              Sign In
            </Link>
          </div>
        ) : (
          <div className="auth-error-box">
            <div className="error-icon">&#x26A0;&#xFE0F;</div>
            <h2>Verification Failed</h2>
            <p>{error}</p>
            <Link to="/login" className="auth-button" style={{ display: 'block', textAlign: 'center', textDecoration: 'none', marginTop: '24px' }}>
              Back to Login
            </Link>
          </div>
        )}
      </div>

      <style>{`
        .auth-page { min-height: 100vh; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; }
        .auth-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 40px; width: 100%; max-width: 400px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); }
        .auth-header { text-align: center; margin-bottom: 32px; }
        .auth-header h1 { color: #fff; font-size: 24px; font-weight: 600; margin: 0; }
        .auth-loading { text-align: center; padding: 40px 0; }
        .loading-spinner { width: 48px; height: 48px; border: 4px solid rgba(255, 255, 255, 0.1); border-top-color: #3b82f6; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 16px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .auth-loading p { color: rgba(255, 255, 255, 0.7); margin: 0; }
        .auth-success { text-align: center; padding: 20px 0; }
        .auth-success .success-icon { font-size: 48px; margin-bottom: 16px; }
        .auth-success h2 { color: #10b981; font-size: 20px; margin: 0 0 16px 0; }
        .auth-success p { color: rgba(255, 255, 255, 0.8); margin: 0; }
        .auth-error-box { text-align: center; padding: 20px 0; }
        .auth-error-box .error-icon { font-size: 48px; margin-bottom: 16px; }
        .auth-error-box h2 { color: #f59e0b; font-size: 20px; margin: 0 0 16px 0; }
        .auth-error-box p { color: rgba(255, 255, 255, 0.7); margin: 0; }
        .auth-button { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: #fff; border: none; border-radius: 8px; padding: 14px 24px; font-size: 16px; font-weight: 600; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }
        .auth-button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); }
      `}</style>
    </div>
  );
}
