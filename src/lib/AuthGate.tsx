/**
 * Auth Gate Component
 *
 * Shows loading spinner while checking auth.
 * Redirects to login when unauthenticated.
 * Shows app content with user info bar when authenticated.
 */

import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from './AuthContext';

interface AuthGateProps {
  children: React.ReactNode;
}

export function AuthGate({ children }: AuthGateProps) {
  const { isAuthenticated, isLoading, user, logout } = useAuth();

  if (isLoading) {
    return (
      <div className="auth-loading">
        <div className="spinner" />
        <p>Loading...</p>

        <style>{`
          .auth-loading {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: rgba(255, 255, 255, 0.6);
            gap: 16px;
          }
          .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(255, 255, 255, 0.2);
            border-top-color: #3b82f6;
            border-radius: 50%;
            animation: spin 1s linear infinite;
          }
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return (
    <div className="auth-wrapper">
      <div className="user-bar">
        <span className="user-email">{user?.email}</span>
        <button className="logout-btn" onClick={logout}>
          Logout
        </button>
      </div>

      {children}

      <style>{`
        .auth-wrapper {
          display: flex;
          flex-direction: column;
          height: 100vh;
        }
        .user-bar {
            height: 32px;
            background: #0f0f1a;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            padding: 0 16px;
            gap: 12px;
            font-size: 12px;
            flex-shrink: 0;
        }
        .user-email {
          color: rgba(255, 255, 255, 0.7);
        }
        .logout-btn {
          margin-left: auto;
          background: none;
          border: 1px solid rgba(255, 255, 255, 0.2);
          color: rgba(255, 255, 255, 0.6);
          padding: 4px 12px;
          border-radius: 4px;
          font-size: 11px;
          cursor: pointer;
          transition: all 0.2s;
        }
        .logout-btn:hover {
          background: rgba(255, 255, 255, 0.1);
          color: #fff;
        }
      `}</style>
    </div>
  );
}
