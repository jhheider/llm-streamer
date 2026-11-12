//! What a failed request looks like to a caller: typed, with the timeout,
//! connection, status and Retry-After information intact. Each test answers
//! from a local TCP listener, so nothing leaves the machine.

use futures::StreamExt;
use llm_streamer::openai::Client;
use llm_streamer::{SecretString, StreamEvent, UpstreamError};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

/// Reads one HTTP request (headers and the Content-Length body) so that the
/// client has finished sending before the server answers or hangs up.
fn read_request(stream: &mut std::net::TcpStream) {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).unwrap_or(0) == 0 {
            return;
        }
        buf.push(byte[0]);
    }
    let headers = String::from_utf8_lossy(&buf).to_lowercase();
    let len: usize = headers
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; len];
    let _ = stream.read_exact(&mut body);
}

/// Serves one connection with `response`, or with silence when `response` is
/// `None` (the client must give up on its own timeout). Returns the base URL.
fn serve_once(response: Option<&'static str>, hold: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}/v1", listener.local_addr().expect("addr"));
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            read_request(&mut stream);
            match response {
                Some(text) => {
                    let _ = stream.write_all(text.as_bytes());
                }
                None => std::thread::sleep(hold),
            }
        }
    });
    base
}

fn client(base: &str, read_timeout: Duration) -> Client {
    // reqwest is built with rustls-no-provider, so a provider must be installed
    // before a client is built. An Err means one is already installed: fine.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .read_timeout(read_timeout)
        .build()
        .expect("client");
    Client::new(base, SecretString::from("sk-test"), "m").with_http_client(http)
}

async fn failure(c: &Client) -> UpstreamError {
    match c.stream_chat(String::new(), vec![]).await {
        Ok(stream) => {
            // A stream that opened is not the failure these tests are about.
            let first = Box::pin(stream).next().await;
            panic!("expected a failed request, got a stream whose first event was {first:?}");
        }
        Err(e) => e,
    }
}

#[tokio::test]
async fn a_silent_server_is_a_timeout_and_the_error_says_so() {
    let base = serve_once(None, Duration::from_secs(3));
    let err = failure(&client(&base, Duration::from_millis(200))).await;
    assert!(matches!(err, UpstreamError::Timeout(_)), "got {err:?}");
    assert!(err.to_string().contains("timed out"), "text: {err}");
}

#[tokio::test]
async fn a_refused_connection_is_a_connect_error() {
    // Bind, learn the port, then close the listener: nothing is listening.
    let port = {
        let l = TcpListener::bind("127.0.0.1:0").expect("bind");
        l.local_addr().expect("addr").port()
    };
    let err = failure(&client(
        &format!("http://127.0.0.1:{port}/v1"),
        Duration::from_secs(5),
    ))
    .await;
    assert!(matches!(err, UpstreamError::Connect(_)), "got {err:?}");
}

#[tokio::test]
async fn a_429_keeps_its_retry_after_and_body() {
    let base = serve_once(
        Some(
            "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 7\r\nContent-Length: 4\r\nConnection: close\r\n\r\nslow",
        ),
        Duration::ZERO,
    );
    let err = failure(&client(&base, Duration::from_secs(5))).await;
    match &err {
        UpstreamError::Status {
            status,
            retry_after,
            body,
        } => {
            assert_eq!(status.as_u16(), 429);
            assert_eq!(*retry_after, Some(Duration::from_secs(7)));
            assert_eq!(body, "slow");
        }
        other => panic!("expected Status, got {other:?}"),
    }
}

#[tokio::test]
async fn a_status_without_retry_after_reads_as_none() {
    let base = serve_once(
        Some(
            "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 4\r\nConnection: close\r\n\r\ndown",
        ),
        Duration::ZERO,
    );
    let err = failure(&client(&base, Duration::from_secs(5))).await;
    assert!(
        matches!(
            err,
            UpstreamError::Status {
                retry_after: None,
                ..
            }
        ),
        "got {err:?}"
    );
}

#[tokio::test]
async fn the_status_wording_callers_match_on_is_unchanged() {
    let base = serve_once(
        Some(
            "HTTP/1.1 429 Too Many Requests\r\nContent-Length: 4\r\nConnection: close\r\n\r\nslow",
        ),
        Duration::ZERO,
    );
    let err = failure(&client(&base, Duration::from_secs(5))).await;
    assert_eq!(err.to_string(), "upstream 429 Too Many Requests: slow");
}

#[tokio::test]
async fn a_stream_that_goes_silent_mid_answer_names_the_timeout() {
    // A good first chunk, then nothing: the read timeout fires in the stream.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let base = format!("http://{}/v1", listener.local_addr().expect("addr"));
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            read_request(&mut stream);
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            );
            let event = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n";
            let _ = stream.write_all(format!("{:x}\r\n{event}\r\n", event.len()).as_bytes());
            let _ = stream.flush();
            std::thread::sleep(Duration::from_secs(3));
        }
    });
    let mut stream = Box::pin(
        client(&base, Duration::from_millis(300))
            .stream_chat(String::new(), vec![])
            .await
            .expect("the answer starts"),
    );
    let mut texts = Vec::new();
    let mut error = None;
    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::TextDelta(t) => texts.push(t),
            StreamEvent::Error(e) => {
                error = Some(e);
                break;
            }
            _ => {}
        }
    }
    assert_eq!(texts, ["partial"]);
    let error = error.expect("the silence must surface as an error event");
    assert!(error.starts_with("stream timed out"), "event: {error}");
}
