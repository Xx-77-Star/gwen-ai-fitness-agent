export async function parseJsonResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const body = await response.text();
  if (!contentType.toLowerCase().includes("application/json")) {
    throw new Error(
      contentType.toLowerCase().includes("text/html")
        ? "接口返回了网页而不是 JSON，请启动 FitLife API 服务后再使用聊天功能。"
        : `接口返回了非 JSON 响应（${contentType || "未知内容类型"}），请检查 API 地址。`
    );
  }
  try {
    return JSON.parse(body);
  } catch {
    throw new Error("接口返回的 JSON 格式无效，请稍后重试。");
  }
}
