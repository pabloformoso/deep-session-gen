"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { register } from "@/lib/api";
import { saveAuth } from "@/lib/auth";

export default function RegisterPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await register(username, email, password);
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
          <h2 className="text-sm tracking-widest text-[#e2e2ff] uppercase">Create Account</h2>

          {[
            { label: "Username", value: username, set: setUsername, type: "text" },
            { label: "Email", value: email, set: setEmail, type: "email" },
            { label: "Password", value: password, set: setPassword, type: "password" },
          ].map(({ label, value, set, type }) => (
            <div key={label}>
              <label className="block text-xs text-muted mb-1 tracking-widest uppercase">{label}</label>
              <input
                type={type}
                value={value}
                onChange={e => set(e.target.value)}
                className="w-full bg-[#0a0a0f] border border-border rounded px-3 py-2 text-sm text-[#e2e2ff] focus:outline-none focus:border-neon transition-colors"
                required
              />
            </div>
          ))}

          {error && <p className="text-danger text-xs">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-neon text-[#0a0a0f] font-bold py-2 rounded text-sm tracking-widest uppercase hover:bg-neon-dim transition-colors disabled:opacity-50"
          >
            {loading ? "..." : "Create Account"}
          </button>

          <p className="text-center text-xs text-muted">
            Have an account?{" "}
            <Link href="/login" className="text-neon hover:underline">
              Sign In
            </Link>
          </p>
        </form>
      </div>
    </div>
  );
}
