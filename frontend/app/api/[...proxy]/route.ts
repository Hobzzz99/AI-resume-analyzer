/**
 * Server-side proxy to the FastAPI service.
 *
 * Every browser request goes to this app's own origin and is forwarded here.
 * Three reasons that is better than calling the service directly from the
 * client:
 *
 * 1. The service address stays server-side. It is not a `NEXT_PUBLIC_` variable,
 *    so it never reaches devtools — and when auth is added later, the credential
 *    is added here rather than shipped to every browser.
 * 2. No CORS preflight. A cross-origin multipart upload triggers an OPTIONS
 *    round trip before every file; same-origin does not.
 * 3. Streaming survives. The SSE response body is piped through untouched, so
 *    stage events still arrive incrementally.
 *
 * The handler is deliberately dumb: it forwards method, body, and status without
 * interpreting them. Any logic here would be a second, divergent copy of the
 * API contract.
 */

import { NextRequest } from "next/server";

const API_BASE_URL = process.env.API_BASE_URL ?? "http://localhost:8000";
const API_PREFIX = process.env.API_PREFIX ?? "/api/v1";

// The proxied request must not be cached or statically evaluated: uploads and
// analyses are mutations, and a cached SSE stream is meaningless.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function targetUrl(request: NextRequest, segments: string[]): string {
  const path = segments.join("/");
  const query = request.nextUrl.search;
  return `${API_BASE_URL}${API_PREFIX}/${path}${query}`;
}

async function forward(
  request: NextRequest,
  segments: string[],
): Promise<Response> {
  const url = targetUrl(request, segments);

  const headers = new Headers();
  // Copied selectively. Forwarding `host` would break virtual-host routing, and
  // forwarding `content-length` after the body is re-streamed can desynchronise
  // it from the actual payload.
  for (const name of ["content-type", "accept", "x-request-id"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }

  const hasBody = request.method !== "GET" && request.method !== "HEAD";

  try {
    const response = await fetch(url, {
      method: request.method,
      headers,
      body: hasBody ? await request.arrayBuffer() : undefined,
      cache: "no-store",
      // Required by undici when streaming a request body.
      // @ts-expect-error -- `duplex` is valid at runtime, missing from the DOM types.
      duplex: "half",
    });

    const responseHeaders = new Headers();
    for (const name of ["content-type", "x-request-id", "cache-control"]) {
      const value = response.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }

    // Body is piped rather than awaited, so an SSE stream stays a stream.
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    // The service being down is the most common failure in local development,
    // so it gets a message that names the actual fix rather than a bare 500.
    const message = error instanceof Error ? error.message : "unknown error";
    return Response.json(
      {
        error: {
          code: "SERVICE_UNREACHABLE",
          message: `Could not reach the analysis service at ${API_BASE_URL}. Is it running? Start it with: uvicorn app.main:app --reload --port 8000`,
          details: { reason: message },
          request_id: null,
        },
      },
      { status: 503 },
    );
  }
}

type RouteContext = { params: Promise<{ proxy: string[] }> };

export async function GET(request: NextRequest, context: RouteContext) {
  return forward(request, (await context.params).proxy);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return forward(request, (await context.params).proxy);
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  return forward(request, (await context.params).proxy);
}
