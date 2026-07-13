import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  // Local dev: forward /api/* to the Python agent proxy (python3 src/agent_server.py).
  // In production on Vercel, /api/* is served by the serverless functions in frontend/api/.
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
