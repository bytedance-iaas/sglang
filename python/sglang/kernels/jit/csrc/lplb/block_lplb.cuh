#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <stdint.h>
namespace {
__device__ int cb(int x, int m) {
  return (x + m - 1) / m;
}
// No Bsum objective. Every accepted move is Pareto-safe against current tails.
template <int P, int R, int M, int I>
__global__ void plan_kernel(const int* gathered, const int* pairs, int K, int* out) {
  __shared__ int q[P], bt[R], tt[R], refb, reft, winx[256], wina[256], winb[256];
  __shared__ long long sk1[256], sk2[256], sk3[256];
  int tid = threadIdx.x;
  for (int p = tid; p < P; p += 256) {
    int s = 0;
    for (int r = 0; r < R; r++)
      s += gathered[r * P + p];
    q[p] = s;
  }
  __syncthreads();
  for (int it = 0; it < I; it++) {
    if (tid < R) {
      int b = 0, t = 0;
      for (int p = tid * (P / R); p < (tid + 1) * (P / R); p++) {
        b += cb(q[p], M);
        t += q[p];
      }
      bt[tid] = b;
      tt[tid] = t;
    }
    __syncthreads();
    int bm = 0, tm = 0;
    for (int r = 0; r < R; r++) {
      bm = max(bm, bt[r]);
      tm = max(tm, tt[r]);
    }
    if (it == 0 && tid == 0) {
      refb = max(1, bm);
      reft = max(1, tm);
    }
    __syncthreads();
    long long best1 = max((long long)bm * reft, (long long)tm * refb),
              best2 = (long long)bm * reft + (long long)tm * refb, best3 = 0;
    for (int r = 0; r < R; r++)
      best3 += (long long)tt[r] * tt[r] + (long long)bt[r] * bt[r] * M * M;
    int ba = -1, bb = -1, bx = 0;
    for (int ci = tid; ci < K * 16; ci += 256) {
      int a = pairs[2 * (ci / 16)], b = pairs[2 * (ci / 16) + 1], ra = a / (P / R), rb = b / (P / R), xa = q[a],
          xb = q[b], n = xa + xb, c = ci % 16;
      int bal = (tt[rb] - xb + n - tt[ra] + xa) / 2, bl = xa + (bt[rb] - bt[ra]) * M / 2;
      int x = c == 0    ? xa
              : c == 1  ? 0
              : c == 2  ? n
              : c == 3  ? bal
              : c == 4  ? bl
              : c == 5  ? (xa / M) * M
              : c == 6  ? cb(xa, M) * M
              : c == 7  ? n - (xb / M) * M
              : c == 8  ? n - cb(xb, M) * M
              : c == 9  ? (bal / M) * M
              : c == 10 ? cb(max(0, bal), M) * M
              : c == 11 ? n - ((n - bal) / M) * M
              : c == 12 ? n - cb(max(0, n - bal), M) * M
              : c == 13 ? (bl / M) * M
              : c == 14 ? cb(max(0, bl), M) * M
                        : bal + 1;
      x = max(0, min(n, x));
      if (ra == rb || x == xa) continue;
      int nb = 0, nt = 0;
      long long sq = 0;
      for (int r = 0; r < R; r++) {
        int t = tt[r], v = bt[r];
        if (r == ra) {
          t += x - xa;
          v += cb(x, M) - cb(xa, M);
        }
        if (r == rb) {
          t += n - x - xb;
          v += cb(n - x, M) - cb(xb, M);
        }
        nb = max(nb, v);
        nt = max(nt, t);
        sq += (long long)t * t + (long long)v * v * M * M;
      }
      if (nb > bm || nt > tm) continue;
      long long s1 = max((long long)nb * reft, (long long)nt * refb), s2 = (long long)nb * reft + (long long)nt * refb;
      if (s1 < best1 || (s1 == best1 && (s2 < best2 || (s2 == best2 && sq < best3)))) {
        best1 = s1;
        best2 = s2;
        best3 = sq;
        ba = a;
        bb = b;
        bx = x;
      }
    }
    sk1[tid] = best1;
    sk2[tid] = best2;
    sk3[tid] = best3;
    wina[tid] = ba;
    winb[tid] = bb;
    winx[tid] = bx;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
      if (tid < stride) {
        int j = tid + stride;
        if (sk1[j] < sk1[tid] ||
            (sk1[j] == sk1[tid] && (sk2[j] < sk2[tid] || (sk2[j] == sk2[tid] && sk3[j] < sk3[tid])))) {
          sk1[tid] = sk1[j];
          sk2[tid] = sk2[j];
          sk3[tid] = sk3[j];
          wina[tid] = wina[j];
          winb[tid] = winb[j];
          winx[tid] = winx[j];
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      int a = wina[0], b = winb[0];
      if (a >= 0) {
        int n = q[a] + q[b];
        q[a] = winx[0];
        q[b] = n - winx[0];
      }
    }
    __syncthreads();
    if (wina[0] < 0) break;
  }
  // Variance is only a search aid; do not reroute if neither tail improved.
  if (tid < R) {
    int b = 0, t = 0;
    for (int p = tid * (P / R); p < (tid + 1) * (P / R); p++) {
      b += cb(q[p], M);
      t += q[p];
    }
    bt[tid] = b;
    tt[tid] = t;
  }
  __syncthreads();
  int finalb = 0, finalt = 0;
  for (int r = 0; r < R; r++) {
    finalb = max(finalb, bt[r]);
    finalt = max(finalt, tt[r]);
  }
  for (int p = tid; p < P; p += 256) {
    int v = q[p];
    if (finalb == refb && finalt == reft) {
      v = 0;
      for (int r = 0; r < R; r++)
        v += gathered[r * P + p];
    }
    out[p] = v;
  }
}
template <int P, int R, int M, int I, int Search>
__global__ void plan_warp_kernel(const int* gathered, const int* pairs, int K, int* out) {
  __shared__ int q[P], bt[R], tt[R], refb, reft, winx[256], wina[256], winb[256];
  __shared__ long long sk1[256], sk2[256], sk3[256];
  int tid = threadIdx.x;
  __shared__ int wx[8], wa[8], wb[8], wp[8];
  __shared__ long long wk1[8], wk2[8], wk3[8];
  for (int p = tid; p < P; p += 256) {
    int s = 0;
    for (int r = 0; r < R; r++)
      s += gathered[r * P + p];
    q[p] = s;
  }
  __syncthreads();
  for (int it = 0; it < I; it++) {
    if constexpr (R == 8) {
      int rank = tid / 32, lane = tid % 32, b = 0, t = 0;
      for (int p = rank * (P / R) + lane; p < (rank + 1) * (P / R); p += 32) {
        b += cb(q[p], M);
        t += q[p];
      }
      for (int off = 16; off > 0; off /= 2) {
        b += __shfl_down_sync(0xffffffff, b, off);
        t += __shfl_down_sync(0xffffffff, t, off);
      }
      if (lane == 0) {
        bt[rank] = b;
        tt[rank] = t;
      }
    } else {
      if (tid < R) {
        int b = 0, t = 0;
        for (int p = tid * (P / R); p < (tid + 1) * (P / R); p++) {
          b += cb(q[p], M);
          t += q[p];
        }
        bt[tid] = b;
        tt[tid] = t;
      }
    }
    __syncthreads();
    int bm = 0, tm = 0;
    for (int r = 0; r < R; r++) {
      bm = max(bm, bt[r]);
      tm = max(tm, tt[r]);
    }
    if (it == 0 && tid == 0) {
      refb = max(1, bm);
      reft = max(1, tm);
    }
    __syncthreads();
    long long best1 = max((long long)bm * reft, (long long)tm * refb),
              best2 = (long long)bm * reft + (long long)tm * refb, best3 = 0;
    for (int r = 0; r < R; r++)
      best3 += (long long)tt[r] * tt[r] + (long long)bt[r] * bt[r] * M * M;
    int ba = -1, bb = -1, bx = 0;
    int candidates = 16;
    if constexpr (Search) {
      int maxn = 0;
      for (int j = 0; j < K; j++)
        maxn = max(maxn, q[pairs[2 * j]] + q[pairs[2 * j + 1]]);
      candidates += 4 * (maxn / M + 1);
    }
    for (int ci = tid; ci < K * candidates; ci += 256) {
      int a = pairs[2 * (ci / candidates)], b = pairs[2 * (ci / candidates) + 1], ra = a / (P / R), rb = b / (P / R),
          xa = q[a], xb = q[b], n = xa + xb, c = ci % candidates;
      int bal = (tt[rb] - xb + n - tt[ra] + xa) / 2, bl = xa + (bt[rb] - bt[ra]) * M / 2;
      int x = c == 0    ? xa
              : c == 1  ? 0
              : c == 2  ? n
              : c == 3  ? bal
              : c == 4  ? bl
              : c == 5  ? (xa / M) * M
              : c == 6  ? cb(xa, M) * M
              : c == 7  ? n - (xb / M) * M
              : c == 8  ? n - cb(xb, M) * M
              : c == 9  ? (bal / M) * M
              : c == 10 ? cb(max(0, bal), M) * M
              : c == 11 ? n - ((n - bal) / M) * M
              : c == 12 ? n - cb(max(0, n - bal), M) * M
              : c == 13 ? (bl / M) * M
              : c == 14 ? cb(max(0, bl), M) * M
                        : bal + 1;
      if constexpr (Search) {
        if (c >= 16) {
          int j = (c - 16) / 4, kind = (c - 16) % 4;
          x = kind == 0 ? j * M : kind == 1 ? j * M + 1 : kind == 2 ? n - j * M : n - j * M - 1;
        }
      }
      x = max(0, min(n, x));
      if (ra == rb || x == xa) continue;
      int nb = 0, nt = 0;
      long long sq = 0;
      for (int r = 0; r < R; r++) {
        int t = tt[r], v = bt[r];
        if (r == ra) {
          t += x - xa;
          v += cb(x, M) - cb(xa, M);
        }
        if (r == rb) {
          t += n - x - xb;
          v += cb(n - x, M) - cb(xb, M);
        }
        nb = max(nb, v);
        nt = max(nt, t);
        sq += (long long)t * t + (long long)v * v * M * M;
      }
      if (nb > bm || nt > tm) continue;
      long long s1 = max((long long)nb * reft, (long long)nt * refb), s2 = (long long)nb * reft + (long long)nt * refb;
      if (s1 < best1 || (s1 == best1 && (s2 < best2 || (s2 == best2 && sq < best3)))) {
        best1 = s1;
        best2 = s2;
        best3 = sq;
        ba = a;
        bb = b;
        bx = x;
      }
    }
    // Explicit original-tree tie priority preserves the reference result.
    int priority = __brev((unsigned)tid) >> 24, lane = tid & 31, warp = tid >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
      long long a1 = __shfl_down_sync(0xffffffff, best1, offset), a2 = __shfl_down_sync(0xffffffff, best2, offset),
                a3 = __shfl_down_sync(0xffffffff, best3, offset);
      int ap = __shfl_down_sync(0xffffffff, priority, offset), aa = __shfl_down_sync(0xffffffff, ba, offset),
          ab = __shfl_down_sync(0xffffffff, bb, offset), ax = __shfl_down_sync(0xffffffff, bx, offset);
      if (lane + offset < 32 &&
          (a1 < best1 ||
           (a1 == best1 && (a2 < best2 || (a2 == best2 && (a3 < best3 || (a3 == best3 && ap < priority))))))) {
        best1 = a1;
        best2 = a2;
        best3 = a3;
        priority = ap;
        ba = aa;
        bb = ab;
        bx = ax;
      }
    }
    if (lane == 0) {
      wk1[warp] = best1;
      wk2[warp] = best2;
      wk3[warp] = best3;
      wp[warp] = priority;
      wa[warp] = ba;
      wb[warp] = bb;
      wx[warp] = bx;
    }
    __syncthreads();
    if (warp == 0) {
      best1 = lane < 8 ? wk1[lane] : 0x7fffffffffffffffLL;
      best2 = lane < 8 ? wk2[lane] : 0x7fffffffffffffffLL;
      best3 = lane < 8 ? wk3[lane] : 0x7fffffffffffffffLL;
      priority = lane < 8 ? wp[lane] : 256;
      ba = lane < 8 ? wa[lane] : -1;
      bb = lane < 8 ? wb[lane] : -1;
      bx = lane < 8 ? wx[lane] : 0;
      for (int offset = 4; offset > 0; offset >>= 1) {
        long long a1 = __shfl_down_sync(0xffffffff, best1, offset), a2 = __shfl_down_sync(0xffffffff, best2, offset),
                  a3 = __shfl_down_sync(0xffffffff, best3, offset);
        int ap = __shfl_down_sync(0xffffffff, priority, offset), aa = __shfl_down_sync(0xffffffff, ba, offset),
            ab = __shfl_down_sync(0xffffffff, bb, offset), ax = __shfl_down_sync(0xffffffff, bx, offset);
        if (lane + offset < 8 &&
            (a1 < best1 ||
             (a1 == best1 && (a2 < best2 || (a2 == best2 && (a3 < best3 || (a3 == best3 && ap < priority))))))) {
          best1 = a1;
          best2 = a2;
          best3 = a3;
          priority = ap;
          ba = aa;
          bb = ab;
          bx = ax;
        }
      }
      if (lane == 0) {
        wina[0] = ba;
        winb[0] = bb;
        winx[0] = bx;
      }
    }
    __syncthreads();
    if (tid == 0) {
      int a = wina[0], b = winb[0];
      if (a >= 0) {
        int n = q[a] + q[b];
        q[a] = winx[0];
        q[b] = n - winx[0];
      }
    }
    __syncthreads();
    if (wina[0] < 0) break;
  }
  // Variance is only a search aid; do not reroute if neither tail improved.
  if constexpr (R == 8) {
    int rank = tid / 32, lane = tid % 32, b = 0, t = 0;
    for (int p = rank * (P / R) + lane; p < (rank + 1) * (P / R); p += 32) {
      b += cb(q[p], M);
      t += q[p];
    }
    for (int off = 16; off > 0; off /= 2) {
      b += __shfl_down_sync(0xffffffff, b, off);
      t += __shfl_down_sync(0xffffffff, t, off);
    }
    if (lane == 0) {
      bt[rank] = b;
      tt[rank] = t;
    }
  } else {
    if (tid < R) {
      int b = 0, t = 0;
      for (int p = tid * (P / R); p < (tid + 1) * (P / R); p++) {
        b += cb(q[p], M);
        t += q[p];
      }
      bt[tid] = b;
      tt[tid] = t;
    }
  }
  __syncthreads();
  int finalb = 0, finalt = 0;
  for (int r = 0; r < R; r++) {
    finalb = max(finalb, bt[r]);
    finalt = max(finalt, tt[r]);
  }
  for (int p = tid; p < P; p += 256) {
    int v = q[p];
    if (finalb == refb && finalt == reft) {
      v = 0;
      for (int r = 0; r < R; r++)
        v += gathered[r * P + p];
    }
    out[p] = v;
  }
}
template <int P, int R, int M, int I, int Search>
void block_plan_warp(tvm::ffi::TensorView counts, tvm::ffi::TensorView pairs, tvm::ffi::TensorView out) {
  using namespace sglang::host;
  SymbolicDevice d;
  SymbolicSize K{"pairs"};
  TensorMatcher({R, P}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(counts);
  TensorMatcher({K, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(pairs);
  TensorMatcher({P}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(out);
  LaunchKernel(1, 256, d.unwrap())(
      plan_warp_kernel<P, R, M, I, Search>,
      (const int*)counts.data_ptr(),
      (const int*)pairs.data_ptr(),
      (int)K.unwrap(),
      (int*)out.data_ptr());
}
template <int P, int R, int M, int I>
void block_plan(tvm::ffi::TensorView counts, tvm::ffi::TensorView pairs, tvm::ffi::TensorView out) {
  using namespace sglang::host;
  SymbolicDevice d;
  SymbolicSize K{"pairs"};
  TensorMatcher({R, P}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(counts);
  TensorMatcher({K, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(pairs);
  TensorMatcher({P}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(out);
  LaunchKernel(1, 256, d.unwrap())(
      plan_kernel<P, R, M, I>,
      (const int*)counts.data_ptr(),
      (const int*)pairs.data_ptr(),
      (int)K.unwrap(),
      (int*)out.data_ptr());
}
template <int P>
__global__ void hist_kernel(const int* ids, int n, int* counts) {
  __shared__ int h[P];
  int t = threadIdx.x;
  for (int p = t; p < P; p += 256)
    h[p] = 0;
  __syncthreads();
  for (int i = t; i < n; i += 256) {
    int p = ids[i];
    if (p >= 0) atomicAdd(h + p, 1);
  }
  __syncthreads();
  for (int p = t; p < P; p += 256)
    counts[p] = h[p];
}
template <int P>
void block_hist(tvm::ffi::TensorView ids, tvm::ffi::TensorView counts) {
  using namespace sglang::host;
  SymbolicDevice d;
  SymbolicSize N{"n"};
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(ids);
  TensorMatcher({P}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(counts);
  LaunchKernel(1, 256, d.unwrap())(
      hist_kernel<P>, (const int*)ids.data_ptr(), (int)N.unwrap(), (int*)counts.data_ptr());
}
template <int P, int R>
__global__ void route_kernel(
    const int* ids, int* out, int n, const int* phy, const int* counts, const int* quota, int* cursor, int rank) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int old = ids[i];
  if (old < 0) {
    out[i] = old;
    return;
  }
  int logical = phy[old];
  bool changed = false;
  int prefix = 0;
  int p = old;
  do {
    int total = 0;
    for (int r = 0; r < R; r++) {
      int v = counts[r * P + p];
      total += v;
      if (r < rank) prefix += v;
    }
    changed |= total != quota[p];
    p = phy[P + p];
  } while (p != old);
  if (!changed) {
    out[i] = old;
    return;
  }
  int ordinal = prefix + atomicAdd(cursor + logical, 1), cum = 0, chosen = old;
  p = old;
  int first = old;
  do {
    first = min(first, p);
    p = phy[P + p];
  } while (p != old);
  p = first;
  do {
    cum += quota[p];
    if (ordinal < cum) {
      chosen = p;
      break;
    }
    p = phy[P + p];
  } while (p != first);
  out[i] = chosen;
}
template <int P, int R>
void block_route(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView phy,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView quota,
    tvm::ffi::TensorView cursor,
    int rank) {
  using namespace sglang::host;
  SymbolicDevice d;
  SymbolicSize N{"n"}, L{"logical"};
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(ids).verify(out);
  TensorMatcher({P}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(quota);
  TensorMatcher({2 * P}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(phy);
  TensorMatcher({R, P}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(counts);
  TensorMatcher({L}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(cursor);
  int n = N.unwrap();
  if (n == 0) return;
  LaunchKernel((n + 255) / 256, 256, d.unwrap())(
      route_kernel<P, R>,
      (const int*)ids.data_ptr(),
      (int*)out.data_ptr(),
      n,
      (const int*)phy.data_ptr(),
      (const int*)counts.data_ptr(),
      (const int*)quota.data_ptr(),
      (int*)cursor.data_ptr(),
      rank);
}
}  // namespace
