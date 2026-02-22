import axios from 'axios';

const API_BASE = typeof __API_BASE__ !== 'undefined' ? __API_BASE__ : 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 300_000, // 5 min for long ops
  headers: {
    'Content-Type': 'application/json',
  },
});

// Session ID interceptor
let currentSessionId: string | null = null;

export function setSessionId(id: string | null) {
  currentSessionId = id;
}

export function getSessionId(): string | null {
  return currentSessionId;
}

apiClient.interceptors.request.use((config) => {
  if (currentSessionId) {
    config.headers['X-Session-ID'] = currentSessionId;
  }
  return config;
});

// Error response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.data?.error) {
      const apiError = error.response.data.error;
      const enhancedError = new Error(apiError.message || 'Unknown error');
      (enhancedError as any).code = apiError.code;
      (enhancedError as any).details = apiError.details;
      (enhancedError as any).status = error.response.status;
      return Promise.reject(enhancedError);
    }
    return Promise.reject(error);
  },
);
