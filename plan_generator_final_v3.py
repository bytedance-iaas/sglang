import sys

plan_str = """1. *Verify paths*
   - Use `run_in_bash_session` with `ls -la rust/sglang-frontend/` to confirm the location of `Cargo.toml`.
2. *Migrate /get_server_info endpoint*
   - Use `replace_with_git_merge_diff` to add `async fn get_server_info(state: State<Arc<AppState>>) -> Json<serde_json::Value>` and `.route("/get_server_info", get(get_server_info))` to `rust/sglang-frontend/src/http_server/mod.rs`. The logic will delegate to `server_info`.
3. *Verify get_server_info implementation*
   - Use `run_in_bash_session` to execute `cargo check --manifest-path rust/sglang-frontend/Cargo.toml` to verify the code compiles.
4. *Migrate /server_info endpoint*
   - Use `replace_with_git_merge_diff` to add `async fn server_info(State(_state): State<Arc<AppState>>) -> Json<serde_json::Value>` and `.route("/server_info", get(server_info))` to `rust/sglang-frontend/src/http_server/mod.rs`. It will return dummy internal state representation as `serde_json::Value`.
5. *Verify server_info implementation*
   - Use `run_in_bash_session` to execute `cargo check --manifest-path rust/sglang-frontend/Cargo.toml` to verify the code compiles.
6. *Migrate /get_weight_version endpoint*
   - Use `replace_with_git_merge_diff` to add `async fn get_weight_version() -> (StatusCode, Json<serde_json::Value>)` and `.route("/get_weight_version", get(get_weight_version))` to `rust/sglang-frontend/src/http_server/mod.rs`. It returns 404 NOT_FOUND.
7. *Verify get_weight_version implementation*
   - Use `run_in_bash_session` to execute `cargo check --manifest-path rust/sglang-frontend/Cargo.toml` to verify the code compiles.
8. *Migrate /weight_version endpoint*
   - Use `replace_with_git_merge_diff` to add `async fn weight_version() -> (StatusCode, Json<serde_json::Value>)` and `.route("/weight_version", get(weight_version))` to `rust/sglang-frontend/src/http_server/mod.rs`. It returns 404 NOT_FOUND.
9. *Verify weight_version implementation*
   - Use `run_in_bash_session` to execute `cargo check --manifest-path rust/sglang-frontend/Cargo.toml` to verify the code compiles.
10. *Migrate /ping endpoint*
    - Use `replace_with_git_merge_diff` to add `async fn sagemaker_health() -> Response` returning `Response::new("".into())` (200 OK) and `.route("/ping", get(sagemaker_health))` to `rust/sglang-frontend/src/http_server/mod.rs`.
11. *Verify ping implementation*
    - Use `run_in_bash_session` to execute `cargo check --manifest-path rust/sglang-frontend/Cargo.toml` to verify the code compiles.
12. *Migrate /get_load endpoint*
    - Use `replace_with_git_merge_diff` to add `async fn get_load(State(_state): State<Arc<AppState>>) -> Json<serde_json::Value>` and `.route("/get_load", get(get_load))` to `rust/sglang-frontend/src/http_server/mod.rs`. It returns deprecated message.
13. *Verify get_load implementation*
    - Use `run_in_bash_session` to execute `cargo check --manifest-path rust/sglang-frontend/Cargo.toml` to verify the code compiles.
14. *Run all tests*
    - Use `run_in_bash_session` to execute `cargo test --manifest-path rust/sglang-frontend/Cargo.toml`.
15. *Complete pre commit steps*
    - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
16. *Submit the change*
    - Submit the code using `submit` tool.
"""

print(plan_str)
