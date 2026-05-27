// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! ZMQ PAIR transport with the same wire framing the Python side uses.
//!
//! Wire from `python/sglang/srt/managers/io_struct/__init__.py`:
//!
//!   * Pickle path: `[b"0xSG01", pickle_bytes]`
//!   * Msgpack path: `[b"0xSG02", msgpack_bytes]`
//!
//! This crate is msgpack-only — the Python `TpWorkerServer` accepts both
//! framings, but the Rust scheduler will never produce pickle.  Receiving
//! a pickle frame is treated as an error (the GPU worker shouldn't be
//! sending those to us if everything is properly typed).

use std::time::Duration;

use crate::wire::Wire;

pub const PICKLE_MAGIC: &[u8] = b"0xSG01";
pub const MSGPACK_MAGIC: &[u8] = b"0xSG02";

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("zmq error: {0}")]
    Zmq(#[from] zmq::Error),

    #[error("malformed frame: expected 2 parts, got {0}")]
    BadFrameCount(usize),

    #[error("unknown magic number on the wire: {0:?}")]
    UnknownMagic(Vec<u8>),

    #[error("received pickle frame; Rust scheduler is msgpack-only")]
    UnexpectedPickle,

    #[error("msgpack encode failed: {0}")]
    Encode(#[from] rmp_serde::encode::Error),

    #[error("msgpack decode failed: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
}

/// PAIR-socket wrapper.  Owns the `zmq::Socket` and adds the same magic-
/// number framing the Python helpers use.
pub struct Transport {
    socket: zmq::Socket,
}

impl Transport {
    /// Connect to a server endpoint (PAIR client side).  The Python
    /// `TpWorkerServer.bind`s; we `connect`.
    pub fn connect(ctx: &zmq::Context, endpoint: &str) -> Result<Self, TransportError> {
        let socket = ctx.socket(zmq::PAIR)?;
        socket.connect(endpoint)?;
        Ok(Self { socket })
    }

    /// Bind to an endpoint (PAIR server side).  Not used by the
    /// scheduler today but provided for symmetry / tests.
    pub fn bind(ctx: &zmq::Context, endpoint: &str) -> Result<Self, TransportError> {
        let socket = ctx.socket(zmq::PAIR)?;
        socket.bind(endpoint)?;
        Ok(Self { socket })
    }

    /// Send a typed wire object as `[MSGPACK_MAGIC, msgpack_bytes]`.
    ///
    /// Mirrors `sock_send` in `io_struct/__init__.py` for the
    /// msgspec.Struct branch.
    pub fn send(&self, obj: &Wire) -> Result<(), TransportError> {
        let bytes = rmp_serde::to_vec_named(obj)?;
        self.socket
            .send_multipart([MSGPACK_MAGIC, &bytes], 0)?;
        Ok(())
    }

    /// Receive one framed wire object.  Blocks until a frame arrives.
    pub fn recv(&self) -> Result<Wire, TransportError> {
        let parts = self.socket.recv_multipart(0)?;
        Self::decode_parts(&parts)
    }

    /// Receive with a timeout (milliseconds).  `Ok(None)` if the timer
    /// elapses with no frame.  Useful for the auto-fallback decode loop.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<Wire>, TransportError> {
        let ms = timeout.as_millis() as i64;
        let mut items = [self.socket.as_poll_item(zmq::POLLIN)];
        let n = zmq::poll(&mut items, ms)?;
        if n == 0 {
            return Ok(None);
        }
        Ok(Some(self.recv()?))
    }

    fn decode_parts(parts: &[Vec<u8>]) -> Result<Wire, TransportError> {
        if parts.len() != 2 {
            return Err(TransportError::BadFrameCount(parts.len()));
        }
        let magic = parts[0].as_slice();
        let body = parts[1].as_slice();
        if magic == MSGPACK_MAGIC {
            Ok(rmp_serde::from_slice::<Wire>(body)?)
        } else if magic == PICKLE_MAGIC {
            Err(TransportError::UnexpectedPickle)
        } else {
            Err(TransportError::UnknownMagic(magic.to_vec()))
        }
    }

    /// Borrow the underlying socket, e.g. for `poll` from a custom loop.
    pub fn socket(&self) -> &zmq::Socket {
        &self.socket
    }
}

/// Receive-only PULL socket for the tokenizer → scheduler stream.
///
/// Mirrors the Python `recv_from_tokenizer` socket: `zmq.PULL`, bound by
/// the scheduler, connected by the tokenizer manager.  Frames use the
/// same `[MAGIC, msgpack_bytes]` shape as the PAIR transport.
pub struct PullSource {
    socket: zmq::Socket,
}

impl PullSource {
    /// Bind a PULL socket to `endpoint` (the scheduler is the listener
    /// in the Python design).
    pub fn bind(ctx: &zmq::Context, endpoint: &str) -> Result<Self, TransportError> {
        let socket = ctx.socket(zmq::PULL)?;
        socket.bind(endpoint)?;
        Ok(Self { socket })
    }

    /// Connect a PULL socket to `endpoint` (rare; useful for tests
    /// when the test driver binds PUSH).
    pub fn connect(ctx: &zmq::Context, endpoint: &str) -> Result<Self, TransportError> {
        let socket = ctx.socket(zmq::PULL)?;
        socket.connect(endpoint)?;
        Ok(Self { socket })
    }

    /// Non-blocking drain — returns every message currently queued on
    /// the socket and stops when `EAGAIN` is hit.  Mirrors the
    /// `while True / NOBLOCK` loop in `Scheduler.recv_requests`.
    pub fn drain_nonblocking(&self) -> Result<Vec<Wire>, TransportError> {
        let mut out = Vec::new();
        loop {
            match self.socket.recv_multipart(zmq::DONTWAIT) {
                Ok(parts) => out.push(Transport::decode_parts(&parts)?),
                Err(zmq::Error::EAGAIN) => break,
                Err(e) => return Err(TransportError::Zmq(e)),
            }
        }
        Ok(out)
    }

    /// Blocking receive — used by tests / single-step harnesses.
    pub fn recv(&self) -> Result<Wire, TransportError> {
        let parts = self.socket.recv_multipart(0)?;
        Transport::decode_parts(&parts)
    }
}

/// Push-side counterpart of `PullSource` — only used by the test
/// harness for now, where Rust plays the role of the tokenizer.
pub struct PushSink {
    socket: zmq::Socket,
}

impl PushSink {
    pub fn connect(ctx: &zmq::Context, endpoint: &str) -> Result<Self, TransportError> {
        let socket = ctx.socket(zmq::PUSH)?;
        socket.connect(endpoint)?;
        Ok(Self { socket })
    }

    pub fn bind(ctx: &zmq::Context, endpoint: &str) -> Result<Self, TransportError> {
        let socket = ctx.socket(zmq::PUSH)?;
        socket.bind(endpoint)?;
        Ok(Self { socket })
    }

    pub fn send(&self, obj: &Wire) -> Result<(), TransportError> {
        let bytes = rmp_serde::to_vec_named(obj)?;
        self.socket
            .send_multipart([MSGPACK_MAGIC, &bytes], 0)?;
        Ok(())
    }
}
