import { getUiSystemPrompt, verifiedUiSchemaLibrary } from "@/lib/verified-ui-contract";

export async function GET() {
  return Response.json({ system_prompt: getUiSystemPrompt(), library_spec: verifiedUiSchemaLibrary.toSpec() }, {
    headers: { "cache-control": "no-store" },
  });
}
