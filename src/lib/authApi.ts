/**
 * Auth API functions for public (unauthenticated) auth endpoints.
 *
 * Uses __API_BASE__ (replaced at build time by Vite).
 */

declare const __API_BASE__: string;
const API_BASE = typeof __API_BASE__ !== 'undefined' && __API_BASE__
  ? __API_BASE__
  : '/api';

async function handleResponse(response: Response): Promise<any> {
  if (!response.ok) {
    const data = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(data.detail || 'Request failed');
  }
  return response.json();
}

export async function forgotPassword(email: string): Promise<void> {
  const response = await fetch(`${API_BASE}/auth/forgot-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  });
  await handleResponse(response);
}

export async function resetPassword(token: string, newPassword: string): Promise<void> {
  const response = await fetch(`${API_BASE}/auth/reset-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token, new_password: newPassword }),
  });
  await handleResponse(response);
}

export async function verifyEmail(token: string): Promise<void> {
  const response = await fetch(`${API_BASE}/auth/verify-email`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  });
  await handleResponse(response);
}

export async function resendVerification(email: string): Promise<void> {
  const response = await fetch(`${API_BASE}/auth/resend-verification`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  });
  await handleResponse(response);
}
