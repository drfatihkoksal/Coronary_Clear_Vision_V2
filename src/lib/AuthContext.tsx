/**
 * Authentication Context and Provider
 *
 * Manages user authentication state across the application.
 * Adapted from v1 for v2's architecture.
 */

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { setSessionId } from './api/client';

// =============================================================================
// Types
// =============================================================================

interface User {
    id: string;
    email: string;
    name: string | null;
    isActive: boolean;
    isVerified: boolean;
    createdAt: string;
}

interface TokenPair {
    accessToken: string;
    refreshToken: string;
    tokenType: string;
    expiresIn: number;
}

interface AuthState {
    user: User | null;
    isLoading: boolean;
    isAuthenticated: boolean;
}

interface AuthContextValue extends AuthState {
    login: (email: string, password: string) => Promise<void>;
    register: (email: string, password: string, name?: string) => Promise<void>;
    logout: () => Promise<void>;
    refreshAuth: () => Promise<boolean>;
}

// =============================================================================
// Constants
// =============================================================================

declare const __API_BASE__: string;
const API_BASE = typeof __API_BASE__ !== 'undefined' && __API_BASE__
  ? __API_BASE__
  : '/api';

const TOKEN_KEY = 'coronary_access_token';
const REFRESH_KEY = 'coronary_refresh_token';

// =============================================================================
// Context
// =============================================================================

const AuthContext = createContext<AuthContextValue | null>(null);

// =============================================================================
// Provider
// =============================================================================

interface AuthProviderProps {
    children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
    const [state, setState] = useState<AuthState>({
        user: null,
        isLoading: true,
        isAuthenticated: false,
    });

    // ---------------------------------------------------------------------------
    // Token Management
    // ---------------------------------------------------------------------------

    const saveTokens = useCallback((tokens: TokenPair) => {
        localStorage.setItem(TOKEN_KEY, tokens.accessToken);
        localStorage.setItem(REFRESH_KEY, tokens.refreshToken);
    }, []);

    const clearTokens = useCallback(() => {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(REFRESH_KEY);
    }, []);

    const getAccessToken = useCallback((): string | null => {
        return localStorage.getItem(TOKEN_KEY);
    }, []);

    const getRefreshToken = useCallback((): string | null => {
        return localStorage.getItem(REFRESH_KEY);
    }, []);

    // ---------------------------------------------------------------------------
    // API Helpers
    // ---------------------------------------------------------------------------

    const fetchWithAuth = useCallback(async (
        url: string,
        options: RequestInit = {}
    ): Promise<Response> => {
        const token = getAccessToken();
        const headers = new Headers(options.headers);

        if (token) {
            headers.set('Authorization', `Bearer ${token}`);
        }

        return fetch(url, { ...options, headers });
    }, [getAccessToken]);

    // ---------------------------------------------------------------------------
    // Auth Actions
    // ---------------------------------------------------------------------------

    const login = useCallback(async (email: string, password: string) => {
        setState(prev => ({ ...prev, isLoading: true }));

        try {
            const response = await fetch(`${API_BASE}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Login failed' }));
                throw new Error(error.detail || 'Login failed');
            }

            const data = await response.json();

            saveTokens({
                accessToken: data.tokens.access_token,
                refreshToken: data.tokens.refresh_token,
                tokenType: data.tokens.token_type,
                expiresIn: data.tokens.expires_in,
            });

            setState({
                user: {
                    id: data.user.id,
                    email: data.user.email,
                    name: data.user.name,
                    isActive: data.user.is_active,
                    isVerified: data.user.is_verified,
                    createdAt: data.user.created_at,
                },
                isLoading: false,
                isAuthenticated: true,
            });
        } catch (error) {
            clearTokens();
            setState({
                user: null,
                isLoading: false,
                isAuthenticated: false,
            });
            throw error;
        }
    }, [saveTokens, clearTokens]);

    const register = useCallback(async (email: string, password: string, name?: string) => {
        setState(prev => ({ ...prev, isLoading: true }));

        try {
            const response = await fetch(`${API_BASE}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, name }),
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Registration failed' }));
                throw new Error(error.detail || 'Registration failed');
            }

            setState(prev => ({ ...prev, isLoading: false }));

        } catch (error) {
            setState(prev => ({ ...prev, isLoading: false }));
            throw error;
        }
    }, []);

    const logout = useCallback(async () => {
        const refreshToken = getRefreshToken();

        if (refreshToken) {
            try {
                await fetch(`${API_BASE}/auth/logout`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ refresh_token: refreshToken }),
                });
            } catch {
                // Ignore logout API errors
            }
        }

        clearTokens();
        setSessionId(null); // Clear session ID on logout
        setState({
            user: null,
            isLoading: false,
            isAuthenticated: false,
        });
    }, [getRefreshToken, clearTokens]);

    const refreshAuth = useCallback(async (): Promise<boolean> => {
        const refreshToken = getRefreshToken();

        if (!refreshToken) {
            return false;
        }

        try {
            const response = await fetch(`${API_BASE}/auth/refresh`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refresh_token: refreshToken }),
            });

            if (!response.ok) {
                throw new Error('Refresh failed');
            }

            const data = await response.json();

            saveTokens({
                accessToken: data.access_token,
                refreshToken: data.refresh_token,
                tokenType: data.token_type,
                expiresIn: data.expires_in,
            });

            return true;
        } catch {
            clearTokens();
            setState({
                user: null,
                isLoading: false,
                isAuthenticated: false,
            });
            return false;
        }
    }, [getRefreshToken, saveTokens, clearTokens]);

    // ---------------------------------------------------------------------------
    // Initialize Auth on Mount
    // ---------------------------------------------------------------------------

    useEffect(() => {
        const initAuth = async () => {
            const token = getAccessToken();

            if (!token) {
                setState(prev => ({ ...prev, isLoading: false }));
                return;
            }

            try {
                const response = await fetchWithAuth(`${API_BASE}/auth/me`);

                if (!response.ok) {
                    const refreshed = await refreshAuth();
                    if (!refreshed) {
                        return;
                    }

                    const retryResponse = await fetchWithAuth(`${API_BASE}/auth/me`);
                    if (!retryResponse.ok) {
                        throw new Error('Auth check failed');
                    }
                }

                const userData = await (await fetchWithAuth(`${API_BASE}/auth/me`)).json();

                setState({
                    user: {
                        id: userData.id,
                        email: userData.email,
                        name: userData.name,
                        isActive: userData.is_active,
                        isVerified: userData.is_verified,
                        createdAt: userData.created_at,
                    },
                    isLoading: false,
                    isAuthenticated: true,
                });
            } catch {
                clearTokens();
                setState({
                    user: null,
                    isLoading: false,
                    isAuthenticated: false,
                });
            }
        };

        initAuth();
    }, [getAccessToken, fetchWithAuth, refreshAuth, clearTokens]);

    // ---------------------------------------------------------------------------
    // Context Value
    // ---------------------------------------------------------------------------

    const value: AuthContextValue = {
        ...state,
        login,
        register,
        logout,
        refreshAuth,
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}

// =============================================================================
// Hook
// =============================================================================

export function useAuth(): AuthContextValue {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}

// =============================================================================
// Utilities
// =============================================================================

export function getStoredAccessToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
}
