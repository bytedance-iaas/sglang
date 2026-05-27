// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! ZMQ transport layer — mirrors `sock_send` / `sock_recv` in
//! `python/sglang/srt/managers/io_struct/__init__.py`.

pub mod zmq_pair;

pub use zmq_pair::{PullSource, PushSink, Transport, TransportError};
