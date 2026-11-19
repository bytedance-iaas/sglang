use openai_protocol::chat::ChatCompletionRequest;
use openai_protocol::common::StreamOptions;
use serde_json::json;
use validator::Validate;

#[test]
fn continuous_usage_and_speculative_metadata_survive_validation_and_forwarding() {
    let body = json!({
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": "Complete real input"}],
        "stream": true,
        "stream_options": {"include_usage": true, "continuous_usage_stats": true},
        "return_spec_tokens_details": true,
        "cache_salt": "preserved-engine-extension",
        "max_tokens": 4096,
        "temperature": 0,
        "top_p": 1
    });
    let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
    request.validate().unwrap();
    let forwarded = serde_json::to_value(&request).unwrap();
    assert_eq!(forwarded["stream_options"]["continuous_usage_stats"], true);
    assert_eq!(forwarded["return_spec_tokens_details"], true);
    assert_eq!(forwarded["cache_salt"], "preserved-engine-extension");
    assert_eq!(forwarded["max_tokens"], 4096);
    assert_eq!(forwarded["temperature"].as_f64(), Some(0.0));
    assert!(!forwarded["ignore_eos"].as_bool().unwrap());
}

#[test]
fn absent_optional_usage_field_keeps_existing_wire_options() {
    let options: StreamOptions = serde_json::from_value(json!({"include_usage": true})).unwrap();
    assert_eq!(
        serde_json::to_value(options).unwrap(),
        json!({"include_usage": true})
    );
}

#[test]
fn standard_validation_still_rejects_empty_messages() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "glm-5.3", "messages": [], "return_spec_tokens_details": true
    }))
    .unwrap();
    assert!(request.validate().is_err());
}
