/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas:          '#0D0F12',
        surface:         '#16181C',
        'surface-raised':'#1E2127',
        border:          '#252830',
        'border-subtle': '#1C1F24',
        'text-primary':  '#E8E8E6',
        'text-secondary':'#8B8D92',
        'text-muted':    '#4B4E56',
        accent:          '#4A9B8E',
        'accent-dim':    '#3A7A70',
        'data-min':      '#1B3F7F',
        'data-cool':     '#3A7EC8',
        'data-zero':     '#6B7077',
        'data-warm':     '#D97220',
        'data-max':      '#B8311F',
      },
      fontFamily: {
        display: ['"Inter Tight"', 'Inter', 'sans-serif'],
        tight:   ['"Inter Tight"', 'Inter', 'sans-serif'],
        sans:    ['Inter', 'sans-serif'],
        mono:    ['"JetBrains Mono"', '"IBM Plex Mono"', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.65rem', { lineHeight: '1rem' }],
      },
      boxShadow: {
        panel:    '0 1px 3px 0 rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.04)',
        'panel-lg':'0 4px 24px 0 rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.05)',
      },
      keyframes: {
        'fade-in':    { from: { opacity: 0, transform: 'translateY(6px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        'pulse-teal': { '0%,100%': { opacity: 1 }, '50%': { opacity: 0.5 } },
      },
      animation: {
        'fade-in':    'fade-in 0.4s ease-out both',
        'pulse-teal': 'pulse-teal 2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
