import assert from "node:assert/strict";
import test from "node:test";
import { parseJsonResponse } from "../frontend/response-json.mjs";

function response(contentType, body) {
  return {
    headers: {get: (name) => name.toLowerCase() === "content-type" ? contentType : null},
    text: async () => body,
  };
}

test("parseJsonResponse parses JSON", async () => {
  assert.deepEqual(
    await parseJsonResponse(response("application/json; charset=utf-8", '{"ok":true}')),
    {ok: true}
  );
});

test("parseJsonResponse reports HTML instead of throwing JSON syntax errors", async () => {
  await assert.rejects(
    parseJsonResponse(response("text/html; charset=utf-8", "<!DOCTYPE html><html>error</html>")),
    /接口返回了网页而不是 JSON/
  );
});

test("parseJsonResponse reports malformed JSON", async () => {
  await assert.rejects(
    parseJsonResponse(response("application/json", "<!DOCTYPE html>")),
    /JSON 格式无效/
  );
});
