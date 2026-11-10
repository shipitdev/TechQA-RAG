/** Next may normalize request.url to localhost even when the browser uses 127.0.0.1. */
export function isAllowedOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  if (!origin) return true;
  try {
    const expected = new URL(process.env.APP_ORIGIN || request.url);
    if (!process.env.APP_ORIGIN) expected.host = request.headers.get("host") || expected.host;
    return new URL(origin).origin === expected.origin;
  } catch {
    return false;
  }
}
