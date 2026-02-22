/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: {
          primary: 'var(--surface-primary)',
          secondary: 'var(--surface-secondary)',
          tertiary: 'var(--surface-tertiary)',
        },
        content: {
          primary: 'var(--content-primary)',
          secondary: 'var(--content-secondary)',
          muted: 'var(--content-muted)',
        },
        brand: 'var(--brand)',
        border: 'var(--border)',
        clinical: {
          normal: 'var(--clinical-normal)',
          intermediate: 'var(--clinical-intermediate)',
          elevated: 'var(--clinical-elevated)',
          'high-risk': 'var(--clinical-high-risk)',
        },
        vessel: {
          lad: '#3b82f6',
          lcx: '#8b5cf6',
          rca: '#ef4444',
          lm: '#f59e0b',
          other: '#6b7280',
          stenosis: '#dc2626',
        },
      },
      width: {
        toolbar: '56px',
        'right-panel': '320px',
      },
      height: {
        header: '48px',
        playback: '64px',
        statusbar: '24px',
      },
      spacing: {
        toolbar: '56px',
        'right-panel': '320px',
        header: '48px',
        playback: '64px',
        statusbar: '24px',
      },
    },
  },
  plugins: [],
};
