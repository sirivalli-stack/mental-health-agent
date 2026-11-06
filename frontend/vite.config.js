import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev/preview servers proxy to the FastAPI backend (Phase 12) so the
// browser always talks same-origin and CORS never enters the picture.
// `/ws` is the Phase 24 voice WebSocket (must forward with ws: true).
const backend = 'http://127.0.0.1:8000'
const proxy = { '/api': backend, '/health': backend, '/ws': { target: backend, ws: true } }

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  test: {
    environment: 'node',
    include: ['test/**/*.test.js'],
  },
})
