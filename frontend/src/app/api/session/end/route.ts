export async function POST(request: Request) {
  const origin = request.headers.get("origin");
  if (origin && origin !== new URL(request.url).origin) return Response.json({ message: "Cross-origin requests are not allowed." }, { status: 403 });
  try {
    const upstream = await fetch("http://127.0.0.1:8000/session/end", {
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
