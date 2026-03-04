import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { checkHealth } from "../api/client";

/**
 * Navbar - Top navigation bar with logo, page links, and server health indicator.
 *
 * The green/red dot pings GET /api/health every 10 seconds so the user
 * always knows if the backend + Docker are running.
 */
export default function Navbar() {
  const location = useLocation();
  const [healthy, setHealthy] = useState(null); // null = checking, true = ok, false = down

  useEffect(() => {
    // Check health immediately on mount, then every 10 seconds
    const check = () => checkHealth().then(setHealthy);
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  const links = [
    { to: "/", label: "Predict" },
    { to: "/benchmark", label: "Benchmark" },
    { to: "/about", label: "About" },
  ];

  return (
    <nav className="bg-slate-800 border-b border-slate-700">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <span className="text-xl"></span>
            <span className="text-lg font-bold text-white">
              HE Benchmark
            </span>
          </Link>

          {/* Navigation links */}
          <div className="flex items-center space-x-1">
            {links.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  location.pathname === link.to
                    ? "bg-slate-700 text-emerald-400"
                    : "text-slate-300 hover:bg-slate-700 hover:text-white"
                }`}
              >
                {link.label}
              </Link>
            ))}

            {/* Health status dot */}
            <div className="ml-4 flex items-center space-x-2">
              <div
                className={`w-2.5 h-2.5 rounded-full ${
                  healthy === null
                    ? "bg-slate-500"
                    : healthy
                    ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]"
                    : "bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]"
                }`}
                title={
                  healthy === null
                    ? "Checking server..."
                    : healthy
                    ? "Server connected"
                    : "Server offline"
                }
              />
              <span className="text-xs text-slate-500">
                {healthy === null ? "..." : healthy ? "Online" : "Offline"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
