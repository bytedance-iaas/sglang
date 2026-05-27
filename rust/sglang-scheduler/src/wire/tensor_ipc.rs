// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! `TensorIPC` — opaque CPU-tensor blob on the wire.
//!
//! Python's `enc_hook` (see `msgpack_struct.py`) encodes any
//! `torch.Tensor` field as the 3-tuple `(shape, dtype_str, raw_bytes)`.
//! On decode, `dec_hook` reconstructs a `torch.Tensor` via
//! `torch.frombuffer(data, dtype=...).reshape(shape)`.
//!
//! The Rust scheduler is CPU-only and never executes tensor ops, so we
//! keep tensors as raw byte blobs plus shape/dtype metadata.  For
//! KV-page bookkeeping (`indices_to_free`, `req_pool_indices`, etc.) we
//! provide typed views via `TensorIPC::as_i64` / `as_i32`.

use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

/// Wire representation of a single `torch.Tensor` field.
///
/// Serialised as a msgpack tuple `(shape, dtype, data)` to match the
/// `enc_hook` in `python/sglang/srt/managers/io_struct/msgpack_struct.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorIPC(
    /// Tensor shape, e.g. `[batch_size]` or `[batch_size, hidden_dim]`.
    pub Vec<i64>,
    /// Dtype string — the suffix of `str(torch.dtype)`, e.g. `"int64"`,
    /// `"int32"`, `"float32"`, `"bfloat16"`.
    pub String,
    /// Raw little-endian bytes of the (contiguous) tensor data.
    pub ByteBuf,
);

impl TensorIPC {
    pub fn shape(&self) -> &[i64] {
        &self.0
    }

    pub fn dtype(&self) -> &str {
        &self.1
    }

    pub fn data(&self) -> &[u8] {
        self.2.as_ref()
    }

    /// Total number of elements (product of shape).
    pub fn numel(&self) -> usize {
        self.0.iter().map(|d| *d as usize).product()
    }

    /// View the bytes as `i64` slice. Returns `None` if dtype mismatch
    /// or the byte buffer isn't 8-aligned in length.
    pub fn as_i64(&self) -> Option<&[i64]> {
        if self.dtype() != "int64" {
            return None;
        }
        let bytes = self.data();
        if bytes.len() != self.numel() * 8 {
            return None;
        }
        // Safety: `int64` tensors arrive packed little-endian with no
        // alignment padding from PyTorch's storage; on a little-endian
        // host (all sglang targets) the cast is sound.
        if self.numel() == 0 {
            return Some(&[]);
        }
        let ptr = bytes.as_ptr() as *const i64;
        Some(unsafe { std::slice::from_raw_parts(ptr, self.numel()) })
    }

    /// View the bytes as `i32` slice. Same conditions as `as_i64`.
    pub fn as_i32(&self) -> Option<&[i32]> {
        if self.dtype() != "int32" {
            return None;
        }
        let bytes = self.data();
        if bytes.len() != self.numel() * 4 {
            return None;
        }
        if self.numel() == 0 {
            return Some(&[]);
        }
        let ptr = bytes.as_ptr() as *const i32;
        Some(unsafe { std::slice::from_raw_parts(ptr, self.numel()) })
    }

    /// Construct from an i64 slice with a 1-D shape.
    pub fn from_i64(values: &[i64]) -> Self {
        let mut buf = Vec::with_capacity(values.len() * 8);
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        TensorIPC(
            vec![values.len() as i64],
            "int64".into(),
            ByteBuf::from(buf),
        )
    }

    /// Construct from an i32 slice with a 1-D shape.
    pub fn from_i32(values: &[i32]) -> Self {
        let mut buf = Vec::with_capacity(values.len() * 4);
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        TensorIPC(
            vec![values.len() as i64],
            "int32".into(),
            ByteBuf::from(buf),
        )
    }

    /// Construct from an i32 slice with a 2-D shape `[rows, cols]`.
    /// Used by the `req_to_token_cpu` snapshot in the model-worker batch.
    pub fn from_i32_2d(values: &[i32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(rows * cols, values.len(), "shape mismatch");
        let mut buf = Vec::with_capacity(values.len() * 4);
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        TensorIPC(
            vec![rows as i64, cols as i64],
            "int32".into(),
            ByteBuf::from(buf),
        )
    }

    /// Construct from an f32 slice with a 1-D shape.
    pub fn from_f32(values: &[f32]) -> Self {
        let mut buf = Vec::with_capacity(values.len() * 4);
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        TensorIPC(
            vec![values.len() as i64],
            "float32".into(),
            ByteBuf::from(buf),
        )
    }

    /// Construct from an f32 slice with a 2-D shape `[rows, cols]`.
    /// Sanity check: `rows * cols == values.len()`.
    pub fn from_f32_2d(values: &[f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(rows * cols, values.len(), "shape mismatch");
        let mut buf = Vec::with_capacity(values.len() * 4);
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        TensorIPC(
            vec![rows as i64, cols as i64],
            "float32".into(),
            ByteBuf::from(buf),
        )
    }

    /// View the bytes as `f32` slice.  Same conditions as `as_i64`.
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype() != "float32" {
            return None;
        }
        let bytes = self.data();
        if bytes.len() != self.numel() * 4 {
            return None;
        }
        let ptr = bytes.as_ptr() as *const f32;
        Some(unsafe { std::slice::from_raw_parts(ptr, self.numel()) })
    }
}
