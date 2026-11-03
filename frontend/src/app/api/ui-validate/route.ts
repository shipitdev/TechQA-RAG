import { inspectUi, type VerifiedSource } from "@/lib/verified-ui-contract";

const MAX_BODY = 128_000;
const MAX_SOURCES = 32;

export async function POST(request: Request) {
  const origin = request.headers.get("origin");
  if (origin && origin !== new URL(request.url).origin) return Response.json({ valid: false, message: "Cross-origin requests are not allowed." }, { status: 403 });
  const length = Number(request.headers.get("content-length") || 0);
  if (length > MAX_BODY) return Response.json({ valid: false, message: "UI payload is too large." }, { status: 413 });
  let raw: string;
  try { raw = await request.text(); } catch { return Response.json({ valid: false, message: "Invalid request." }, { status: 400 }); }
  if (raw.length > MAX_BODY) return Response.json({ valid: false, message: "UI payload is too large." }, { status: 413 });
  let body: unknown;
  try { body = JSON.parse(raw); } catch { return Response.json({ valid: false, message: "Request body must be valid JSON." }, { status: 400 }); }
  if (!body || typeof body !== "object") return Response.json({ valid: false, message: "Invalid request." }, { status: 400 });

  const { openui, sources } = body as { openui?: unknown; sources?: unknown };
  if (typeof openui !== "string" || !openui.trim() || openui.length > 64_000 || !Array.isArray(sources) || sources.length > MAX_SOURCES) {
    return Response.json({ valid: false, message: "OpenUI and its source list are required." }, { status: 400 });
  }
  if (!sources.every((source) => {
    if (!source || typeof source !== "object") return false;
    const record = source as Record<string, unknown>;
    return typeof record.id === "string" && typeof record.title === "string" && typeof record.text === "string" && (record.url === undefined || record.url === null || typeof record.url === "string");
  })) {
    return Response.json({ valid: false, message: "Source records are malformed." }, { status: 400 });
  }

  if (openui.length > 32_000) return Response.json({ valid: false, message: "OpenUI content is too large." }, { status: 413 });
  let result;
  try { result = inspectUi(openui, sources as VerifiedSource[]); }
  catch { return Response.json({ valid: false, message: "OpenUI content could not be parsed safely." }, { status: 422 }); }
  return Response.json(result, { status: result.valid ? 200 : 422 });
}
