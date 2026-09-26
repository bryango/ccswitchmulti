//! Chat-level hosted tool loop.

use super::{
    image_generation::{
        self, error_tool_content as image_error_tool_content,
        result_to_tool_content as image_result_to_tool_content, HostedImageGenerationConfig,
        IMAGE_GENERATION_FUNCTION_NAME,
    },
    openai_client::OpenAiHostedToolClient,
    web_search::{
        self, error_tool_content, parse_arguments, query_hash, result_to_tool_content,
        HostedWebSearchConfig, WEB_SEARCH_FUNCTION_NAME,
    },
};
use serde_json::{json, Value};

pub(crate) const HOSTED_TOOL_LOOP_HEADER: &str = "x-cc-switch-hosted-tool-loop";
pub(crate) const MAX_HOSTED_TOOL_ITERATIONS: usize = 3;

/// Codex 入站 hosted tools 的本地执行配置。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct HostedToolLoopConfig {
    pub(crate) web_search: Option<HostedWebSearchConfig>,
    pub(crate) image_generation: Option<HostedImageGenerationConfig>,
}

impl HostedToolLoopConfig {
    pub(crate) fn is_empty(&self) -> bool {
        self.web_search.is_none() && self.image_generation.is_none()
    }
}

/// 已桥接的 hosted tool 类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostedToolCallKind {
    WebSearch,
    ImageGeneration,
}

impl HostedToolCallKind {
    fn from_function_name(name: &str) -> Option<Self> {
        match name {
            WEB_SEARCH_FUNCTION_NAME => Some(Self::WebSearch),
            IMAGE_GENERATION_FUNCTION_NAME => Some(Self::ImageGeneration),
            _ => None,
        }
    }
}

/// 第三方 Chat response 中的一个 hosted tool call。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostedToolCall {
    pub(crate) kind: HostedToolCallKind,
    pub(crate) id: String,
    pub(crate) arguments: String,
}

/// Chat tool-call 扫描结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HostedToolCallScan {
    NoToolCalls,
    OnlyHosted(Vec<HostedToolCall>),
    ContainsUnsupportedToolCalls,
}

/// Native Responses tool ownership classification.
///
/// CCSM may continue the request only when every actionable client-side call in
/// the response belongs to the local hosted-tool bridge. Mixed ownership cannot
/// be safely split across the local continuation and Codex's own tool executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResponsesHostedToolCallScan {
    NoToolCalls,
    OnlyHosted(Vec<HostedToolCall>),
    OnlyClientToolCalls,
    MixedHostedAndClientToolCalls,
    InvalidHostedToolCall,
}

pub(crate) fn project_hosted_tools_for_responses_request(
    request: &mut Value,
    web_search_enabled: bool,
) -> HostedToolLoopConfig {
    let mut config = HostedToolLoopConfig::default();
    let mut saw_hosted_web_search = false;
    let mut saw_hosted_image_generation = false;

    if let Some(tools) = request.get_mut("tools").and_then(Value::as_array_mut) {
        let mut projected = Vec::with_capacity(tools.len());
        for tool in tools.drain(..) {
            match tool.get("type").and_then(Value::as_str) {
                Some("web_search") => {
                    saw_hosted_web_search = true;
                    if web_search_enabled {
                        config.web_search = Some(web_search::config_from_tool(&tool));
                        projected.push(web_search::responses_tool_definition());
                    }
                }
                Some("image_generation") => {
                    // Native third-party Responses does not yet have a local image
                    // bridge. Omit the hosted declaration instead of forwarding a
                    // tool the upstream may reject or silently ignore.
                    saw_hosted_image_generation = true;
                }
                _ => projected.push(tool),
            }
        }
        *tools = projected;
    }

    rewrite_responses_hosted_tool_choice(
        request,
        config.web_search.is_some(),
        saw_hosted_web_search,
        saw_hosted_image_generation,
    );

    if request
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty)
    {
        request.as_object_mut().unwrap().remove("tools");
    }

    config
}

fn rewrite_responses_hosted_tool_choice(
    request: &mut Value,
    web_search_projected: bool,
    saw_hosted_web_search: bool,
    saw_hosted_image_generation: bool,
) {
    let Some(choice) = request.get_mut("tool_choice") else {
        return;
    };

    let choice_type = choice.get("type").and_then(Value::as_str);
    match choice_type {
        Some("web_search") if saw_hosted_web_search => {
            if web_search_projected {
                *choice = json!({
                    "type": "function",
                    "name": web_search::WEB_SEARCH_FUNCTION_NAME
                });
            } else {
                *choice = json!("auto");
            }
        }
        Some("image_generation") if saw_hosted_image_generation => {
            *choice = json!("auto");
        }
        Some("allowed_tools") => {
            let Some(allowed) = choice.get_mut("tools").and_then(Value::as_array_mut) else {
                return;
            };
            let mut rewritten = Vec::with_capacity(allowed.len());
            for tool in allowed.drain(..) {
                match tool.get("type").and_then(Value::as_str) {
                    Some("web_search") if saw_hosted_web_search => {
                        if web_search_projected {
                            rewritten.push(json!({
                                "type": "function",
                                "name": web_search::WEB_SEARCH_FUNCTION_NAME
                            }));
                        }
                    }
                    Some("image_generation") if saw_hosted_image_generation => {}
                    _ => rewritten.push(tool),
                }
            }
            *allowed = rewritten;
            if allowed.is_empty() {
                *choice = json!("auto");
            }
        }
        _ => {}
    }
}

/// A forced hosted-tool choice applies only to the first model round.
///
/// Once CCSM has executed that hosted tool and appended its output, the model
/// must be allowed to answer, request another search, or hand a client-owned
/// tool back to Codex instead of being deterministically forced to search again.
pub(crate) fn relax_hosted_tool_choice_for_responses_request(
    request: &mut Value,
    config: &HostedToolLoopConfig,
) {
    let Some(choice) = request.get("tool_choice").cloned() else {
        return;
    };

    let should_relax = match &choice {
        Value::String(value) => value == "required",
        Value::Object(object) => match object.get("type").and_then(Value::as_str) {
            Some("function") => object
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| hosted_function_is_enabled(name, config)),
            Some("allowed_tools") => {
                object.get("mode").and_then(Value::as_str) == Some("required")
                    && object
                        .get("tools")
                        .and_then(Value::as_array)
                        .is_some_and(|tools| {
                            tools
                                .iter()
                                .any(|tool| responses_tool_selector_is_enabled_hosted(tool, config))
                        })
            }
            _ => false,
        },
        _ => false,
    };
    if !should_relax {
        return;
    }

    if choice.get("type").and_then(Value::as_str) == Some("allowed_tools") {
        request["tool_choice"]["mode"] = json!("auto");
    } else {
        request["tool_choice"] = json!("auto");
    }
}

fn hosted_function_is_enabled(name: &str, config: &HostedToolLoopConfig) -> bool {
    match HostedToolCallKind::from_function_name(name) {
        Some(HostedToolCallKind::WebSearch) => config.web_search.is_some(),
        Some(HostedToolCallKind::ImageGeneration) => config.image_generation.is_some(),
        None => false,
    }
}

fn responses_tool_selector_is_enabled_hosted(tool: &Value, config: &HostedToolLoopConfig) -> bool {
    match tool.get("type").and_then(Value::as_str) {
        Some("function") => tool
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|name| hosted_function_is_enabled(name, config)),
        Some("web_search") => config.web_search.is_some(),
        Some("image_generation") => config.image_generation.is_some(),
        _ => false,
    }
}

pub(crate) fn scan_responses_hosted_tool_calls(
    response: &Value,
    config: &HostedToolLoopConfig,
) -> ResponsesHostedToolCallScan {
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return ResponsesHostedToolCallScan::NoToolCalls;
    };

    let mut calls = Vec::new();
    let mut saw_client_tool = false;

    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => {
                let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                let kind = HostedToolCallKind::from_function_name(name);
                let is_enabled = match kind {
                    Some(HostedToolCallKind::WebSearch) => config.web_search.is_some(),
                    Some(HostedToolCallKind::ImageGeneration) => config.image_generation.is_some(),
                    None => false,
                };
                if !is_enabled {
                    saw_client_tool = true;
                    continue;
                }
                let Some(kind) = kind else {
                    saw_client_tool = true;
                    continue;
                };
                let id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .or_else(|| item.get("id").and_then(Value::as_str))
                    .unwrap_or("")
                    .to_string();
                if id.is_empty() {
                    return ResponsesHostedToolCallScan::InvalidHostedToolCall;
                }
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}")
                    .to_string();
                calls.push(HostedToolCall {
                    kind,
                    id,
                    arguments,
                });
            }
            Some(item_type) if item_type.ends_with("_call") => {
                // Any non-hosted call-like item belongs to the client. This is
                // intentionally fail-safe for current shell/MCP/computer/file
                // calls and future Responses call types that CCSM does not know.
                saw_client_tool = true;
            }
            _ => {}
        }
    }

    if calls.is_empty() {
        return if saw_client_tool {
            ResponsesHostedToolCallScan::OnlyClientToolCalls
        } else {
            ResponsesHostedToolCallScan::NoToolCalls
        };
    }
    if saw_client_tool {
        ResponsesHostedToolCallScan::MixedHostedAndClientToolCalls
    } else {
        ResponsesHostedToolCallScan::OnlyHosted(calls)
    }
}

pub(crate) fn normalize_responses_input_for_hosted_continuation(request: &mut Value) -> bool {
    match request.get("input").cloned() {
        None | Some(Value::Null) => {
            request["input"] = json!([]);
            true
        }
        Some(Value::Array(_)) => true,
        Some(Value::Object(item)) => {
            request["input"] = Value::Array(vec![Value::Object(item)]);
            true
        }
        Some(Value::String(text)) => {
            request["input"] = json!([{
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": text
                }]
            }]);
            true
        }
        _ => false,
    }
}

pub(crate) fn remove_projected_hosted_function_calls_from_responses_response(
    response: &mut Value,
    config: &HostedToolLoopConfig,
) -> usize {
    let Some(output) = response.get_mut("output").and_then(Value::as_array_mut) else {
        return 0;
    };
    let before = output.len();
    output.retain(|item| {
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            return true;
        }
        let Some(name) = item.get("name").and_then(Value::as_str) else {
            return true;
        };
        !hosted_function_is_enabled(name, config)
    });
    before.saturating_sub(output.len())
}

pub(crate) fn disable_projected_hosted_functions_for_responses_request(
    request: &mut Value,
    config: &HostedToolLoopConfig,
) {
    let remove_tools = if let Some(tools) = request.get_mut("tools").and_then(Value::as_array_mut) {
        tools.retain(|tool| {
            if tool.get("type").and_then(Value::as_str) != Some("function") {
                return true;
            }
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                return true;
            };
            !hosted_function_is_enabled(name, config)
        });
        tools.is_empty()
    } else {
        false
    };
    if remove_tools {
        request.as_object_mut().unwrap().remove("tools");
    }
    request["tool_choice"] = json!("auto");
}

pub(crate) fn hosted_tool_error_messages(calls: &[HostedToolCall], message: &str) -> Vec<Value> {
    calls
        .iter()
        .map(|call| {
            json!({
                "role": "tool",
                "tool_call_id": call.id.clone(),
                "content": json!({"error": message}).to_string()
            })
        })
        .collect()
}

pub(crate) fn append_tool_outputs_to_responses_request(
    request: &mut Value,
    response: &Value,
    tool_messages: Vec<Value>,
) -> bool {
    if !normalize_responses_input_for_hosted_continuation(request) {
        return false;
    }
    let Some(input) = request.get_mut("input").and_then(Value::as_array_mut) else {
        return false;
    };
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return false;
    };

    let hosted_call_ids = tool_messages
        .iter()
        .filter_map(|message| message.get("tool_call_id").and_then(Value::as_str))
        .filter(|call_id| !call_id.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning" | "message") => input.push(item.clone()),
            Some("function_call") => {
                let replay_call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .filter(|call_id| !call_id.is_empty())
                    .or_else(|| item.get("id").and_then(Value::as_str));
                let Some(replay_call_id) = replay_call_id else {
                    continue;
                };
                if !hosted_call_ids
                    .iter()
                    .any(|call_id| call_id == replay_call_id)
                {
                    continue;
                }

                let mut replay = item.clone();
                if replay
                    .get("call_id")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
                {
                    replay["call_id"] = Value::String(replay_call_id.to_string());
                }
                input.push(replay);
            }
            _ => {}
        }
    }
    for message in tool_messages {
        let call_id = message
            .get("tool_call_id")
            .and_then(Value::as_str)
            .unwrap_or("");
        let content = message
            .get("content")
            .cloned()
            .unwrap_or(Value::String(String::new()));
        if call_id.is_empty() {
            return false;
        }
        input.push(json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": content
        }));
    }
    if let Some(obj) = request.as_object_mut() {
        obj.insert("stream".to_string(), json!(false));
    }
    true
}

/// 扫描 Chat response 是否只请求了本地可执行的 hosted tools。
pub(crate) fn scan_hosted_tool_calls(chat_response: &Value) -> HostedToolCallScan {
    let Some(message) = first_choice_message(chat_response) else {
        return HostedToolCallScan::NoToolCalls;
    };

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        if tool_calls.is_empty() {
            return HostedToolCallScan::NoToolCalls;
        }
        let mut calls = Vec::new();
        for (index, tool_call) in tool_calls.iter().enumerate() {
            let function = tool_call.get("function").unwrap_or(&Value::Null);
            let name = function.get("name").and_then(Value::as_str).unwrap_or("");
            let Some(kind) = HostedToolCallKind::from_function_name(name) else {
                return HostedToolCallScan::ContainsUnsupportedToolCalls;
            };
            let id = tool_call
                .get("id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("call_{index}"));
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}")
                .to_string();
            calls.push(HostedToolCall {
                kind,
                id,
                arguments,
            });
        }
        return HostedToolCallScan::OnlyHosted(calls);
    }

    if let Some(function_call) = message.get("function_call") {
        let name = function_call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("");
        if name.is_empty() {
            return HostedToolCallScan::NoToolCalls;
        }
        let Some(kind) = HostedToolCallKind::from_function_name(name) else {
            return HostedToolCallScan::ContainsUnsupportedToolCalls;
        };
        let arguments = function_call
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("{}")
            .to_string();
        return HostedToolCallScan::OnlyHosted(vec![HostedToolCall {
            kind,
            id: "call_0".to_string(),
            arguments,
        }]);
    }

    HostedToolCallScan::NoToolCalls
}

/// 将 assistant tool-call message 与 tool output messages 追加到 Chat 请求体。
///
/// 返回:
/// - `true` 表示成功追加；`false` 表示缺少 messages 或 assistant message。
///
/// 副作用:
/// - 修改 `chat_request.messages`，并确保后续请求为非流式。
pub(crate) fn append_tool_outputs_to_chat_request(
    chat_request: &mut Value,
    chat_response: &Value,
    tool_messages: Vec<Value>,
) -> bool {
    let Some(assistant_message) = first_choice_message(chat_response).cloned() else {
        return false;
    };
    let Some(messages) = chat_request
        .get_mut("messages")
        .and_then(Value::as_array_mut)
    else {
        return false;
    };

    messages.push(assistant_message);
    messages.extend(tool_messages);
    if let Some(obj) = chat_request.as_object_mut() {
        obj.insert("stream".to_string(), json!(false));
        obj.remove("stream_options");
    }
    true
}

/// 执行一组 hosted tool calls 并生成 Chat tool messages。
pub(crate) async fn execute_hosted_tool_calls(
    calls: &[HostedToolCall],
    config: &HostedToolLoopConfig,
    client: &Result<OpenAiHostedToolClient, String>,
    trace_id: Option<&str>,
) -> Vec<Value> {
    let mut messages = Vec::new();

    for call in calls {
        let content = match call.kind {
            HostedToolCallKind::WebSearch => {
                execute_web_search_call(client, call, config.web_search.as_ref(), trace_id).await
            }
            HostedToolCallKind::ImageGeneration => {
                execute_image_generation_call(
                    client,
                    call,
                    config.image_generation.as_ref(),
                    trace_id,
                )
                .await
            }
        };

        messages.push(json!({
            "role": "tool",
            "tool_call_id": call.id,
            "content": content
        }));
    }

    messages
}

async fn execute_web_search_call(
    client: &Result<OpenAiHostedToolClient, String>,
    call: &HostedToolCall,
    config: Option<&HostedWebSearchConfig>,
    trace_id: Option<&str>,
) -> String {
    let Some(config) = config else {
        return error_tool_content(
            &call.arguments,
            "web_search hosted tool is not configured for this request",
        );
    };
    let args = parse_arguments(&call.arguments);
    let hash = query_hash(&args.query);
    let started = std::time::Instant::now();
    match client {
        Ok(client) if !args.query.trim().is_empty() => {
            match client.run_web_search(&args, config).await {
                Ok(result) => {
                    log_hosted_tool_event(
                        trace_id,
                        WEB_SEARCH_FUNCTION_NAME,
                        &hash,
                        "ok",
                        started.elapsed().as_millis(),
                        None,
                    );
                    result_to_tool_content(&result)
                }
                Err(err) => {
                    let message = safe_error_message(&err.to_string());
                    log_hosted_tool_event(
                        trace_id,
                        WEB_SEARCH_FUNCTION_NAME,
                        &hash,
                        "error",
                        started.elapsed().as_millis(),
                        Some(&message),
                    );
                    error_tool_content(&args.query, &message)
                }
            }
        }
        Ok(_) => {
            let message = "web_search query is empty";
            log_hosted_tool_event(
                trace_id,
                WEB_SEARCH_FUNCTION_NAME,
                &hash,
                "invalid",
                started.elapsed().as_millis(),
                Some(message),
            );
            error_tool_content(&args.query, message)
        }
        Err(err) => {
            let message = safe_error_message(err);
            log_hosted_tool_event(
                trace_id,
                WEB_SEARCH_FUNCTION_NAME,
                &hash,
                "not_configured",
                started.elapsed().as_millis(),
                Some(&message),
            );
            error_tool_content(&args.query, &message)
        }
    }
}

async fn execute_image_generation_call(
    client: &Result<OpenAiHostedToolClient, String>,
    call: &HostedToolCall,
    config: Option<&HostedImageGenerationConfig>,
    trace_id: Option<&str>,
) -> String {
    let Some(config) = config else {
        return image_error_tool_content(
            &call.arguments,
            "image_generation hosted tool is not configured for this request",
        );
    };
    let args = image_generation::parse_arguments(&call.arguments, config);
    let hash = image_generation::prompt_hash(&args.prompt);
    let started = std::time::Instant::now();
    match client {
        Ok(client) if !args.prompt.trim().is_empty() => {
            match client.run_image_generation(&args, config).await {
                Ok(result) => {
                    log_hosted_tool_event(
                        trace_id,
                        IMAGE_GENERATION_FUNCTION_NAME,
                        &hash,
                        "ok",
                        started.elapsed().as_millis(),
                        None,
                    );
                    image_result_to_tool_content(&result)
                }
                Err(err) => {
                    let message = safe_error_message(&err.to_string());
                    log_hosted_tool_event(
                        trace_id,
                        IMAGE_GENERATION_FUNCTION_NAME,
                        &hash,
                        "error",
                        started.elapsed().as_millis(),
                        Some(&message),
                    );
                    image_error_tool_content(&args.prompt, &message)
                }
            }
        }
        Ok(_) => {
            let message = "image_generation prompt is empty";
            log_hosted_tool_event(
                trace_id,
                IMAGE_GENERATION_FUNCTION_NAME,
                &hash,
                "invalid",
                started.elapsed().as_millis(),
                Some(message),
            );
            image_error_tool_content(&args.prompt, message)
        }
        Err(err) => {
            let message = safe_error_message(err);
            log_hosted_tool_event(
                trace_id,
                IMAGE_GENERATION_FUNCTION_NAME,
                &hash,
                "not_configured",
                started.elapsed().as_millis(),
                Some(&message),
            );
            image_error_tool_content(&args.prompt, &message)
        }
    }
}

/// 取第一条 Chat choice message。
fn first_choice_message(chat_response: &Value) -> Option<&Value> {
    chat_response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
}

/// 写入 hosted tool 脱敏诊断事件。
fn log_hosted_tool_event(
    trace_id: Option<&str>,
    tool: &str,
    query_hash: &str,
    status: &str,
    elapsed_ms: u128,
    error: Option<&str>,
) {
    if let Some(trace_id) = trace_id {
        let mut fields = vec![
            ("trace", trace_id.to_string()),
            ("tool", tool.to_string()),
            ("query_hash", query_hash.to_string()),
            ("status", status.to_string()),
            ("elapsed_ms", elapsed_ms.to_string()),
        ];
        if let Some(error) = error {
            fields.push(("error", error.to_string()));
        }
        crate::proxy::codex_router_log::append_event("hosted_tool_call", &fields);
    }
}

/// 写入“hosted tool 已投影但上游没有发起调用”的脱敏诊断事件。
///
/// 该事件只在调用方已经明确要求某个 hosted tool 时写入；普通
/// `tool_choice=auto` 下模型自然选择不搜索不应被标记为故障。
pub(crate) fn log_hosted_tool_not_called(
    trace_id: Option<&str>,
    session: &str,
    model: &str,
    provider: &str,
    tool: &str,
    streaming: bool,
) {
    let Some(trace_id) = trace_id else {
        return;
    };
    crate::proxy::codex_router_log::append_event(
        "hosted_tool_not_called",
        &[
            ("trace", trace_id.to_string()),
            ("session", session.to_string()),
            ("model", model.to_string()),
            ("provider", provider.to_string()),
            ("tool", tool.to_string()),
            ("status", "not_called".to_string()),
            (
                "reason",
                "upstream_returned_success_without_hosted_tool_call".to_string(),
            ),
            ("streaming", streaming.to_string()),
        ],
    );
}

/// 裁剪错误文本，避免把上游长响应或敏感上下文回填给模型。
fn safe_error_message(message: &str) -> String {
    let normalized = message.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= 500 {
        return normalized;
    }
    let mut truncated = normalized.chars().take(500).collect::<String>();
    truncated.push_str("...");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosted_tool_loop_config_is_empty_only_when_no_tools() {
        assert!(HostedToolLoopConfig::default().is_empty());
        assert!(!HostedToolLoopConfig {
            web_search: Some(HostedWebSearchConfig::default()),
            ..HostedToolLoopConfig::default()
        }
        .is_empty());
        assert!(!HostedToolLoopConfig {
            image_generation: Some(HostedImageGenerationConfig::default()),
            ..HostedToolLoopConfig::default()
        }
        .is_empty());
    }

    #[test]
    fn scan_hosted_tool_calls_accepts_web_search_and_image_generation() {
        let response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_search",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": "{\"query\":\"Codex\"}"
                            }
                        },
                        {
                            "id": "call_image",
                            "type": "function",
                            "function": {
                                "name": "generate_image",
                                "arguments": "{\"prompt\":\"robot\"}"
                            }
                        }
                    ]
                }
            }]
        });

        assert_eq!(
            scan_hosted_tool_calls(&response),
            HostedToolCallScan::OnlyHosted(vec![
                HostedToolCall {
                    kind: HostedToolCallKind::WebSearch,
                    id: "call_search".to_string(),
                    arguments: "{\"query\":\"Codex\"}".to_string()
                },
                HostedToolCall {
                    kind: HostedToolCallKind::ImageGeneration,
                    id: "call_image".to_string(),
                    arguments: "{\"prompt\":\"robot\"}".to_string()
                }
            ])
        );
    }

    #[test]
    fn scan_hosted_tool_calls_rejects_mixed_tools() {
        let response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_file",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{}"
                        }
                    }]
                }
            }]
        });

        assert_eq!(
            scan_hosted_tool_calls(&response),
            HostedToolCallScan::ContainsUnsupportedToolCalls
        );
    }

    #[test]
    fn project_hosted_web_search_to_responses_function() {
        let mut request = json!({
            "tools": [
                {"type":"web_search","external_web_access":true,"search_content_types":["text"]},
                {"type":"function","name":"shell","parameters":{"type":"object"}}
            ]
        });

        let config = project_hosted_tools_for_responses_request(&mut request, true);

        assert!(config.web_search.is_some());
        assert_eq!(request["tools"][0]["type"], "function");
        assert_eq!(request["tools"][0]["name"], "web_search");
        assert_eq!(request["tools"][1]["name"], "shell");
    }

    #[test]
    fn project_explicit_web_search_choice_to_function_choice() {
        let mut request = json!({
            "tools": [{"type":"web_search"}],
            "tool_choice": {"type":"web_search"}
        });

        let config = project_hosted_tools_for_responses_request(&mut request, true);

        assert!(config.web_search.is_some());
        assert_eq!(request["tool_choice"]["type"], "function");
        assert_eq!(request["tool_choice"]["name"], "web_search");
    }

    #[test]
    fn project_disabled_native_hosted_tools_are_omitted() {
        let mut request = json!({
            "tools": [
                {"type":"web_search"},
                {"type":"image_generation"},
                {"type":"function","name":"shell","parameters":{"type":"object"}}
            ],
            "tool_choice": {"type":"web_search"}
        });

        let config = project_hosted_tools_for_responses_request(&mut request, false);

        assert!(config.is_empty());
        assert_eq!(request["tools"].as_array().unwrap().len(), 1);
        assert_eq!(request["tools"][0]["name"], "shell");
        assert_eq!(request["tool_choice"], "auto");
    }

    #[test]
    fn project_allowed_tools_rewrites_hosted_search_and_drops_unowned_image() {
        let mut request = json!({
            "tools": [
                {"type":"web_search"},
                {"type":"image_generation"},
                {"type":"function","name":"shell","parameters":{"type":"object"}}
            ],
            "tool_choice": {
                "type":"allowed_tools",
                "mode":"auto",
                "tools":[
                    {"type":"web_search"},
                    {"type":"image_generation"},
                    {"type":"function","name":"shell"}
                ]
            }
        });

        let config = project_hosted_tools_for_responses_request(&mut request, true);

        assert!(config.web_search.is_some());
        assert_eq!(request["tool_choice"]["type"], "allowed_tools");
        assert_eq!(request["tool_choice"]["tools"].as_array().unwrap().len(), 2);
        assert_eq!(request["tool_choice"]["tools"][0]["type"], "function");
        assert_eq!(request["tool_choice"]["tools"][0]["name"], "web_search");
        assert_eq!(request["tool_choice"]["tools"][1]["name"], "shell");
    }

    #[test]
    fn project_empty_allowed_tools_downgrades_to_auto() {
        let mut request = json!({
            "tools": [{"type":"web_search"}],
            "tool_choice": {
                "type":"allowed_tools",
                "mode":"auto",
                "tools":[{"type":"web_search"}]
            }
        });

        let config = project_hosted_tools_for_responses_request(&mut request, false);

        assert!(config.is_empty());
        assert!(request.get("tools").is_none());
        assert_eq!(request["tool_choice"], "auto");
    }

    #[test]
    fn relax_required_choice_after_hosted_round() {
        let mut request = json!({"tool_choice":"required"});
        let config = HostedToolLoopConfig {
            web_search: Some(HostedWebSearchConfig::default()),
            image_generation: None,
        };

        relax_hosted_tool_choice_for_responses_request(&mut request, &config);

        assert_eq!(request["tool_choice"], "auto");
    }

    #[test]
    fn relax_required_allowed_tools_after_hosted_round() {
        let mut request = json!({
            "tool_choice": {
                "type":"allowed_tools",
                "mode":"required",
                "tools":[
                    {"type":"function","name":"web_search"},
                    {"type":"function","name":"shell"}
                ]
            }
        });
        let config = HostedToolLoopConfig {
            web_search: Some(HostedWebSearchConfig::default()),
            image_generation: None,
        };

        relax_hosted_tool_choice_for_responses_request(&mut request, &config);

        assert_eq!(request["tool_choice"]["mode"], "auto");
        assert_eq!(request["tool_choice"]["tools"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn normalize_responses_hosted_input_wraps_object_and_absent() {
        let mut object_request = json!({
            "input": {"role":"user","content":[{"type":"input_text","text":"hello"}]}
        });
        assert!(normalize_responses_input_for_hosted_continuation(
            &mut object_request
        ));
        assert_eq!(object_request["input"].as_array().unwrap().len(), 1);
        assert_eq!(object_request["input"][0]["role"], "user");

        let mut absent_request = json!({});
        assert!(normalize_responses_input_for_hosted_continuation(
            &mut absent_request
        ));
        assert_eq!(absent_request["input"], json!([]));
    }

    #[test]
    fn strip_projected_hosted_calls_preserves_client_calls() {
        let config = HostedToolLoopConfig {
            web_search: Some(HostedWebSearchConfig::default()),
            image_generation: None,
        };
        let mut response = json!({
            "output": [
                {"type":"function_call","call_id":"search","name":"web_search","arguments":"{}"},
                {"type":"custom_tool_call","call_id":"patch","name":"apply_patch","input":"x"}
            ]
        });

        assert_eq!(
            remove_projected_hosted_function_calls_from_responses_response(&mut response, &config),
            1
        );
        assert_eq!(response["output"].as_array().unwrap().len(), 1);
        assert_eq!(response["output"][0]["type"], "custom_tool_call");
    }

    #[test]
    fn append_responses_hosted_output_normalizes_string_input() {
        let mut request = json!({"input":"hello"});
        let response = json!({
            "output":[{
                "type":"function_call",
                "call_id":"call_search",
                "name":"web_search",
                "arguments":"{}"
            }]
        });

        assert!(append_tool_outputs_to_responses_request(
            &mut request,
            &response,
            vec![json!({
                "role":"tool",
                "tool_call_id":"call_search",
                "content":"{}"
            })],
        ));

        let input = request["input"]
            .as_array()
            .expect("normalized Responses input");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "hello");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[2]["type"], "function_call_output");
    }

    #[test]
    fn scan_responses_hosted_web_search_call() {
        let config = HostedToolLoopConfig {
            web_search: Some(HostedWebSearchConfig::default()),
            image_generation: None,
        };
        let response = json!({
            "output": [{
                "type":"function_call",
                "call_id":"call_search",
                "name":"web_search",
                "arguments":"{\"query\":\"Codex\"}"
            }]
        });

        assert_eq!(
            scan_responses_hosted_tool_calls(&response, &config),
            ResponsesHostedToolCallScan::OnlyHosted(vec![HostedToolCall {
                kind: HostedToolCallKind::WebSearch,
                id: "call_search".to_string(),
                arguments: "{\"query\":\"Codex\"}".to_string()
            }])
        );
    }

    #[test]
    fn scan_responses_hosted_calls_detects_mixed_client_tool_ownership() {
        let config = HostedToolLoopConfig {
            web_search: Some(HostedWebSearchConfig::default()),
            image_generation: None,
        };
        for client_item in [
            json!({
                "type":"function_call",
                "call_id":"call_shell",
                "name":"shell",
                "arguments":"{}"
            }),
            json!({
                "type":"custom_tool_call",
                "call_id":"call_patch",
                "name":"apply_patch",
                "input":"*** Begin Patch"
            }),
            json!({
                "type":"tool_search_call",
                "call_id":"call_tool_search",
                "arguments":"{}"
            }),
            json!({
                "type":"local_shell_call",
                "call_id":"call_shell",
                "command":"pwd"
            }),
            json!({
                "type":"mcp_tool_call",
                "call_id":"call_mcp",
                "name":"lookup"
            }),
            json!({
                "type":"future_tool_call",
                "call_id":"call_future"
            }),
        ] {
            let response = json!({
                "output": [
                    {
                        "type":"function_call",
                        "call_id":"call_search",
                        "name":"web_search",
                        "arguments":"{\"query\":\"Codex\"}"
                    },
                    client_item
                ]
            });

            assert_eq!(
                scan_responses_hosted_tool_calls(&response, &config),
                ResponsesHostedToolCallScan::MixedHostedAndClientToolCalls
            );
        }
    }

    #[test]
    fn scan_responses_client_tool_only_is_pass_through() {
        let config = HostedToolLoopConfig {
            web_search: Some(HostedWebSearchConfig::default()),
            image_generation: None,
        };
        let response = json!({
            "output": [{
                "type":"custom_tool_call",
                "call_id":"call_patch",
                "name":"apply_patch",
                "input":"*** Begin Patch"
            }]
        });

        assert_eq!(
            scan_responses_hosted_tool_calls(&response, &config),
            ResponsesHostedToolCallScan::OnlyClientToolCalls
        );
    }

    #[test]
    fn relax_responses_hosted_tool_choice_after_first_round() {
        let mut request = json!({
            "tool_choice": {"type":"function","name":"web_search"}
        });
        let config = HostedToolLoopConfig {
            web_search: Some(HostedWebSearchConfig::default()),
            image_generation: None,
        };

        relax_hosted_tool_choice_for_responses_request(&mut request, &config);

        assert_eq!(request["tool_choice"], "auto");
    }

    #[test]
    fn append_responses_hosted_output_does_not_replay_client_owned_calls() {
        let mut request = json!({"input":[]});
        let response = json!({
            "output": [
                {
                    "type":"custom_tool_call",
                    "call_id":"call_patch",
                    "name":"apply_patch",
                    "input":"*** Begin Patch"
                },
                {
                    "type":"tool_search_call",
                    "call_id":"call_tool_search",
                    "arguments":"{}"
                },
                {
                    "type":"function_call",
                    "call_id":"call_search",
                    "name":"web_search",
                    "arguments":"{}"
                }
            ]
        });

        assert!(append_tool_outputs_to_responses_request(
            &mut request,
            &response,
            vec![json!({
                "role":"tool",
                "tool_call_id":"call_search",
                "content":"{}"
            })],
        ));

        let input = request["input"].as_array().expect("responses input");
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[0]["call_id"], "call_search");
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "call_search");
    }

    #[test]
    fn append_responses_hosted_output_replays_call_and_result() {
        let mut request = json!({
            "input":[{"role":"user","content":[{"type":"input_text","text":"Search."}]}]
        });
        let response = json!({
            "output":[{
                "type":"function_call",
                "id":"fc_1",
                "call_id":"call_search",
                "name":"web_search",
                "arguments":"{\"query\":\"Codex\"}"
            }]
        });

        assert!(append_tool_outputs_to_responses_request(
            &mut request,
            &response,
            vec![json!({
                "role":"tool",
                "tool_call_id":"call_search",
                "content":"{\"summary\":\"ok\"}"
            })],
        ));

        assert_eq!(request["input"][1]["type"], "function_call");
        assert_eq!(request["input"][2]["type"], "function_call_output");
        assert_eq!(request["input"][2]["call_id"], "call_search");
        assert_eq!(request["stream"], false);
    }

    #[test]
    fn append_responses_hosted_output_injects_fallback_call_id() {
        let mut request = json!({"input":[]});
        let response = json!({
            "output":[{
                "type":"function_call",
                "id":"fc_search",
                "name":"web_search",
                "arguments":"{}"
            }]
        });

        assert!(append_tool_outputs_to_responses_request(
            &mut request,
            &response,
            vec![json!({
                "role":"tool",
                "tool_call_id":"fc_search",
                "content":"{}"
            })],
        ));

        assert_eq!(request["input"][0]["call_id"], "fc_search");
        assert_eq!(request["input"][1]["call_id"], "fc_search");
    }

    #[test]
    fn append_tool_outputs_to_chat_request_adds_assistant_and_tool_messages() {
        let mut request = json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Generate."}],
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_image",
                        "type": "function",
                        "function": {"name": "generate_image", "arguments": "{}"}
                    }]
                }
            }]
        });

        assert!(append_tool_outputs_to_chat_request(
            &mut request,
            &response,
            vec![json!({
                "role": "tool",
                "tool_call_id": "call_image",
                "content": "{}"
            })],
        ));
        assert_eq!(request["messages"].as_array().unwrap().len(), 3);
        assert_eq!(request["stream"], false);
        assert!(request.get("stream_options").is_none());
    }
}
