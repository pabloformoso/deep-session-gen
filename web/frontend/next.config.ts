import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // E2E runs set NEXT_DIST_DIR=.next-e2e so the mock-mode dev server's lockfile
  // lives in a separate build dir and does not collide with the engineer's
  // running `npm run dev` session (Next 16 refuses two dev servers per project).
  distDir: process.env.NEXT_DIST_DIR || ".next",
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.APOLLO_API_URL ?? "http://localhost:8000"}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
