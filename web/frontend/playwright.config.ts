import { defineConfig, devices } from "@playwright/test";
import path from "path";

const PROJECT_ROOT = path.resolve(__dirname, "../..");
const E2E_DB = path.join(PROJECT_ROOT, ".tmp/apollo-e2e.db");

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",

  use: {
    baseURL: "http://localhost:3001",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],

  webServer: [
    {
      command: `rm -f "${E2E_DB}" && mkdir -p "${path.dirname(E2E_DB)}" && uv run uvicorn backend.app:app --port 8801 --app-dir web`,
      cwd: PROJECT_ROOT,
      url: "http://localhost:8801/docs",
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
      env: {
        AGENT_PROVIDER: "mock",
        APOLLO_DB_PATH: E2E_DB,
        APOLLO_CORS_ORIGINS: "http://localhost:3001,http://127.0.0.1:3001",
        JWT_SECRET: "e2e-secret",
        // The health URL hits a 401 without auth, which Playwright treats as up.
      },
      ignoreHTTPSErrors: true,
    },
    {
      command: "npm run dev -- --port 3001",
      cwd: __dirname,
      url: "http://localhost:3001",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
      env: {
        NEXT_PUBLIC_API_BASE: "http://localhost:8801",
        NEXT_PUBLIC_WS_BASE: "ws://localhost:8801",
        NEXT_DIST_DIR: ".next-e2e",
      },
    },
  ],
});
