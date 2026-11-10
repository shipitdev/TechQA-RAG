import { isAllowedOrigin } from "@/lib/request-origin";

export async function POST(request: Request) {
  if (!isAllowedOrigin(request)) return Response.json({ message: "Cross-origin requests are not allowed." }, { status: 403 });
  if (Number(request.headers.get("content-length") || 0) > 16_000) return Response.json({ message: "Feedback is too large." }, { status: 413 });
  let body: string;
  try { body = await request.text(); }
  catch { return Response.json({ message: "Feedback could not be read." }, { status: 400 }); }
  if (body.length > 16_000) return Response.json({ message: "Feedback is too large." }, { status: 413 });
  try {
    const upstream = await fetch(new URL("feedback", process.env.RAG_API_URL || "http://127.0.0.1:8000/ask"), {
      method: "POST",
      headers: { "content-type": "application/json", ...(request.headers.get("cookie") ? { cookie: request.headers.get("cookie")! } : {}) },
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    return new Response(upstream.body, { status: upstream.status, headers: { "content-type": "application/json", "cache-control": "no-store" } });
  } catch {
    return Response.json({ message: "Feedback could not be saved. Please retry." }, { status: 502 });
  }
}
