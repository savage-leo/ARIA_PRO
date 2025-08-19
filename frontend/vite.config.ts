import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    proxy: {
      '/trading': { target: 'http://localhost:8000', changeOrigin: true },
      '/account': { target: 'http://localhost:8000', changeOrigin: true },
      '/market': { target: 'http://localhost:8000', changeOrigin: true },
      '/positions': { target: 'http://localhost:8000', changeOrigin: true },
      '/signals': { target: 'http://localhost:8000', changeOrigin: true },
      '/monitoring': { target: 'http://localhost:8000', changeOrigin: true },
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws': { target: 'ws://localhost:8000', ws: true }
    }
  }
})
