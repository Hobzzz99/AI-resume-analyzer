/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // The API base URL is read server-side only, inside the proxy route handler.
  // It is deliberately NOT a NEXT_PUBLIC_ variable: keeping it off the client
  // means the browser only ever talks to its own origin, so the service address
  // (and any credential added later) never reaches a user's devtools, and file
  // uploads incur no CORS preflight.
  env: {},

  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
