//! Rate limiting integration tests
//!
//! Tests for rate limiting and concurrency control.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod rate_limiting_tests {
    use super::*;

    async fn send_generate(app: axum::Router, label: &str) -> StatusCode {
        let payload = json!({
            "text": label,
            "stream": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        app.oneshot(req).await.unwrap().status()
    }

    /// Test that concurrent requests are handled within limits
    #[tokio::test]
    async fn test_concurrent_requests_within_limit() {
        let config = TestRouterConfig::with_concurrency(3400, 10);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(19300, 50)]).await;

        let app = ctx.create_app().await;

        // Send 5 concurrent requests (within limit of 10)
        let mut handles = Vec::new();
        let success_count = Arc::new(AtomicUsize::new(0));

        for i in 0..5 {
            let app_clone = app.clone();
            let success_clone = Arc::clone(&success_count);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Concurrent request {}", i),
                    "stream": false
                });

                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();

                let resp = app_clone.oneshot(req).await.unwrap();
                if resp.status() == StatusCode::OK {
                    success_clone.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        // Wait for all requests to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // All requests should succeed within the limit
        assert_eq!(
            success_count.load(Ordering::SeqCst),
            5,
            "All concurrent requests should succeed within limit"
        );

        ctx.shutdown().await;
    }

    /// Test rate limit tokens per second
    #[tokio::test]
    async fn test_rate_limit_tokens() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3401)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(100)
            .rate_limit_tokens_per_second(50) // 50 tokens/sec
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(19301)]).await;

        let app = ctx.create_app().await;

        // Send requests and verify they succeed
        let mut success_count = 0;
        for i in 0..10 {
            let payload = json!({
                "text": format!("Rate limited request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        // Most requests should succeed (some might be rate limited)
        assert!(
            success_count >= 5,
            "At least half of requests should succeed, got {}",
            success_count
        );

        ctx.shutdown().await;
    }

    /// Test unlimited concurrent requests when set to 0
    #[tokio::test]
    async fn test_unlimited_concurrent_requests() {
        let config = TestRouterConfig::with_concurrency(3402, 0); // Unlimited

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(19302, 10)]).await;

        let app = ctx.create_app().await;

        // Send many concurrent requests
        let mut handles = Vec::new();
        let success_count = Arc::new(AtomicUsize::new(0));

        for i in 0..20 {
            let app_clone = app.clone();
            let success_clone = Arc::clone(&success_count);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Unlimited request {}", i),
                    "stream": false
                });

                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();

                let resp = app_clone.oneshot(req).await.unwrap();
                if resp.status() == StatusCode::OK {
                    success_clone.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // All requests should succeed with unlimited concurrency
        assert_eq!(
            success_count.load(Ordering::SeqCst),
            20,
            "All requests should succeed with unlimited concurrency"
        );

        ctx.shutdown().await;
    }

    /// Test queue behavior when requests exceed capacity
    #[tokio::test]
    async fn test_queue_behavior() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3403)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(2) // Very low limit
            .queue_size(5) // Small queue
            .queue_timeout_secs(1) // Short timeout
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::slow(19303, 200)], // Longer delay to test queuing
        )
        .await;

        let app = ctx.create_app().await;

        // Send requests that will exceed capacity
        let mut handles = Vec::new();
        let success_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));

        for i in 0..10 {
            let app_clone = app.clone();
            let success_clone = Arc::clone(&success_count);
            let error_clone = Arc::clone(&error_count);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Queue test {}", i),
                    "stream": false
                });

                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();

                let resp = app_clone.oneshot(req).await.unwrap();
                if resp.status() == StatusCode::OK {
                    success_clone.fetch_add(1, Ordering::SeqCst);
                } else {
                    error_clone.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // Some requests should succeed, some might timeout or be rejected
        let successes = success_count.load(Ordering::SeqCst);
        let errors = error_count.load(Ordering::SeqCst);

        assert!(
            successes > 0,
            "At least some requests should succeed, got {} successes and {} errors",
            successes,
            errors
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_strict_concurrency_and_queue_full_statuses() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3404)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(10)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(1)
            .rate_limit_tokens_per_second(0)
            .queue_size(1)
            .queue_timeout_secs(2)
            .build_unchecked();
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(19304, 400)]).await;
        let app = ctx.create_app().await;

        let first = tokio::spawn(send_generate(app.clone(), "active"));
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        let second = tokio::spawn(send_generate(app.clone(), "queued"));
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        let third_status = send_generate(app, "rejected").await;

        assert_eq!(third_status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(first.await.unwrap(), StatusCode::OK);
        assert_eq!(second.await.unwrap(), StatusCode::OK);
        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_queue_timeout_is_distinct_from_request_timeout() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3405)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(10)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(1)
            .rate_limit_tokens_per_second(0)
            .queue_size(1)
            .queue_timeout_secs(1)
            .build_unchecked();
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(19305, 1_500)])
                .await;
        let app = ctx.create_app().await;

        let first = tokio::spawn(send_generate(app.clone(), "active"));
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        let queued_status = send_generate(app, "times out in queue").await;

        assert_eq!(queued_status, StatusCode::REQUEST_TIMEOUT);
        assert_eq!(first.await.unwrap(), StatusCode::OK);
        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_cancelled_waiter_releases_queue_capacity_immediately() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3406)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(10)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(1)
            .rate_limit_tokens_per_second(0)
            .queue_size(1)
            .queue_timeout_secs(2)
            .build_unchecked();
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(19306, 500)]).await;
        let app = ctx.create_app().await;

        let active = tokio::spawn(send_generate(app.clone(), "active"));
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        let cancelled_waiter = tokio::spawn(send_generate(app.clone(), "cancelled waiter"));
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        cancelled_waiter.abort();
        assert!(cancelled_waiter.await.unwrap_err().is_cancelled());

        let replacement = tokio::time::timeout(
            tokio::time::Duration::from_secs(2),
            send_generate(app, "replacement waiter"),
        )
        .await
        .expect("replacement waiter should not hang");

        assert_eq!(replacement, StatusCode::OK);
        assert_eq!(active.await.unwrap(), StatusCode::OK);
        ctx.shutdown().await;
    }
}
