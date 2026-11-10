export async function GET() {
  try {
    const upstream = new URL("health/ready", process.env.RAG_API_URL || "http://127.0.0.1:8000/ask");
    const response = await fetch(upstream, { cache: "no-store", signal: AbortSignal.timeout(5_000) });
    if (!response.ok) throw new Error("Readiness check failed");
    const body = await response.json() as { presentation_status?: unknown; presentation?: unknown };
    const presentation = body.presentation_status ?? body.presentation;
    const presentationStatus = typeof presentation === "string" ? presentation : "unknown";
    return Response.json({ api: "ready", presentation_status: presentationStatus });
  } catch {
    return Response.json({ api: "unavailable", presentation_status: "unknown" }, { status: 200 });
  }
}
