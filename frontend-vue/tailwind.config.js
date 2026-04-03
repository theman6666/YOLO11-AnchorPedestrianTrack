/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Manual control via .dark class on HTML element
  theme: {
    extend: {
      colors: {
        // Industrial dark mode palette
        gray: {
          850: '#1f2937',
          900: '#111827',
          950: '#030712',
        },
        // Accent colors for industrial feel
        accent: {
          50: '#eff6ff',
          100: '#dbeafe',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        }
      }
    }
  },
  plugins: []
}
