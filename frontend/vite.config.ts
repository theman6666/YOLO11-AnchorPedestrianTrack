import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
  ],

  // Path aliases for clean imports
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },

  // Development server configuration
  server: {
    port: 5173,
    host: true, // Listen on all addresses
    strictPort: false, // Automatically try next available port

    // Proxy API requests to Flask backend
    proxy: {
      // Proxy video feed endpoint
      '/video_feed': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
        ws: true // Enable WebSocket proxy if needed
      },

      // Proxy detection endpoints
      '/detect': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false
      },

      // Proxy result files
      '/results': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false
      }
    }
  }
})
