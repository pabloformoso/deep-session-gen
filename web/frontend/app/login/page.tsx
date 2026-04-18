"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { login } from "@/lib/api";
import { saveAuth } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await login(username, password);
      saveAuth(res.access_token, res.user);
      router.push("/dashboard");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <h1 className="font-pixel text-neon text-lg glow tracking-widest">APOLLO</h1>
          <p className="text-muted text-xs mt-2 tracking-widest">AGENTS v2.0</p>
        </div>

        <form onSubmit={handleSubmit} className="bg-surface border border-border rounded p-6 space-y-4">
          <h2 className="text-sm tracking-widest text-[#e2e2ff] uppercase">Sign In</h2>

          <div>
            <label className="block text-xs text-muted mb-1 tracking-widest uppercase">Username</label>
            <input
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              className="w-full bg-[#0a0a0f] border border-border rounded px-3 py-2 text-sm text-[#e2e2ff] focus:outline-none focus:border-neon transition-colors"
              required
              autoFocus
            />
          </div>

          <div>
            <label className="block text-xs text-muted mb-1 tracking-widest uppercase">Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full bg-[#0a0a0f] border border-border rounded px-3 py-2 text-sm text-[#e2e2ff] focus:outline-none focus:border-neon transition-colors"
              required
            />
          </div>

          {error && (
            <p className="text-danger text-xs">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-neon text-[#0a0a0f] font-bold py-2 rounded text-sm tracking-widest uppercase hover:bg-neon-dim transition-colors disabled:opacity-50"
          >
            {loading ? "..." : "Enter"}
          </button>

          <p className="text-center text-xs text-muted">
            No account?{" "}
            <Link href="/register" className="text-neon hover:underline">
              Register
            </Link>
          </p>
        </form>
      </div>
    </div>
  );
}
