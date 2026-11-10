import { isAllowedOrigin } from "@/lib/request-origin";
const MAX_QUESTION_LENGTH = 8_000;

export async function POST(request: Request) {
  if (!isAllowedOrigin(request)) return Response.json({ message: "Cross-origin requests are not allowed." }, { status: 403 });
  if (Number(request.headers.get("content-length") || 0) > 64_000) return Response.json({ message: "Request body is too large." }, { status: 413 });
  let input: unknown;
  try {
    input = await request.json();
  } catch {
    return Response.json({ message: "Request body must be valid JSON." }, { status: 400 });
  }

  if (!input || typeof input !== "object") return Response.json({ message: "A question is required." }, { status: 400 });
  const { question, regenerate } = input as { question?: unknown; regenerate?: unknown };
  if (typeof question !== "string" || !question.trim() || question.length > MAX_QUESTION_LENGTH) {
    return Response.json({ message: `Enter a question of ${MAX_QUESTION_LENGTH.toLocaleString()} characters or fewer.` }, { status: 400 });
  }
  if (regenerate !== undefined && typeof regenerate !== "boolean") {
    return Response.json({ message: "The regenerate flag must be a boolean." }, { status: 400 });
  }

  try {
    const upstream = await fetch(process.env.RAG_API_URL || "http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: {
        accept: "text/event-stream",
        "content-type": "application/json",
        ...(request.headers.get("cookie") ? { cookie: request.headers.get("cookie")! } : {}),
      },
      body: JSON.stringify({ question, ...(regenerate ? { regenerate: true } : {}) }),
      cache: "no-store",
      signal: request.signal,
    });

    const headers = new Headers();
    for (const name of ["content-type", "cache-control", "x-request-id"]) {
      const value = upstream.headers.get(name);
      if (value) headers.set(name, value);
    }
    headers.set("cache-control", "no-cache, no-transform");
    headers.set("x-accel-buffering", "no");
    for (const cookie of upstream.headers.getSetCookie()) headers.append("set-cookie", cookie);
    return new Response(upstream.body, { status: upstream.status, headers });
  } catch {
    if (request.signal.aborted) return new Response(null, { status: 499 });
    return Response.json({ message: "The research service is unavailable. Check that the local API is running." }, { status: 502 });
  }
}
