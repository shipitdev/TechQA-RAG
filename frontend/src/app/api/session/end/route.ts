import { isAllowedOrigin } from "@/lib/request-origin";
export async function POST(request: Request) {
  if (!isAllowedOrigin(request)) return Response.json({ message: "Cross-origin requests are not allowed." }, { status: 403 });
  try {
    const upstream = await fetch(new URL("session/end", process.env.RAG_API_URL || "http://127.0.0.1:8000/ask"), {
      method: "POST",
      headers: request.headers.get("cookie") ? { cookie: request.headers.get("cookie")! } : {},
      cache: "no-store",
    });
    const headers = new Headers();
    const contentType = upstream.headers.get("content-type");
    if (contentType) headers.set("content-type", contentType);
    for (const cookie of upstream.headers.getSetCookie()) headers.append("set-cookie", cookie);
    return new Response(upstream.ok ? null : upstream.body, { status: upstream.status, headers });
  } catch {
    return Response.json({ message: "The research service is unavailable." }, { status: 502 });
  }
}
