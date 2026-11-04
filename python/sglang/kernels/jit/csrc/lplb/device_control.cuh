#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "wan22_shape.cuh"
#include <limits.h>
#include <stdint.h>
namespace {
using DCV = tvm::ffi::TensorView;
#ifndef WAN22_BALANCE_BLOCK_SIZE
#define WAN22_BALANCE_BLOCK_SIZE 128
#endif
static_assert(WAN22_BALANCE_BLOCK_SIZE == 64 || WAN22_BALANCE_BLOCK_SIZE == 128);
__device__ int dc_cb(int x) {
  return (x + WAN22_BALANCE_BLOCK_SIZE - 1) / WAN22_BALANCE_BLOCK_SIZE;
}
__global__ void dc_prepare_kernel(const int64_t* m, const int* q, int* w) {
  __shared__ int mapping[DP], ns[DL], totals[DL], rt[DR], rb[DR];
  int t = threadIdx.x;
  for (int e = t; e < DL; e += blockDim.x) {
    ns[e] = 0;
    totals[e] = 0;
  }
  if (t < DR) {
    rt[t] = 0;
    rb[t] = 0;
  }
  for (int p = t; p < DP; p += 256)
    mapping[p] = m[p];
  __syncthreads();
  for (int p = t; p < DP; p += 256) {
    int e = mapping[p];
    atomicAdd(ns + e, 1);
    if (q) {
      int v = q[p];
      atomicAdd(totals + e, v);
      atomicAdd(rt + p / DS, v);
      atomicAdd(rb + p / DS, dc_cb(v));
    }
  }
  __syncthreads();
  for (int e = t; e < DL; e += blockDim.x) {
    w[NS + e] = ns[e];
    w[LT + e] = totals[e];
  }
  if (t < DR) {
    w[TT + t] = rt[t];
    w[BB + t] = rb[t];
  }
  for (int p = t; p < DP; p += 256) {
    int e = mapping[p], ix = 0;
    if (ns[e] > 1)
      for (int j = 0; j < p; j++)
        ix += mapping[j] == e;
    w[SL + e * DC + ix] = p;
  }
  if (t == 0) {
    int n = 0;
    for (int p = 0; p < DP; p++)
      if (ns[mapping[p]] > 1) w[RP + n++] = p;
    w[NR] = n;
  }
}
__device__ int dc_cap(int r, int B, int T, const int* b, const int* t) {
  return min((B - b[r]) * WAN22_BALANCE_BLOCK_SIZE, max(0, T - t[r]));
}
__global__ void dc_candidates_kernel(const int64_t* m, const int* q, const int* w, int64_t* keys) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= DN) return;
  int64_t* key = keys + idx * 8;
  key[0] = INT64_MAX;
  int hot = idx / DREP, k = idx % DREP;
  if (hot >= DL || w[NS + hot] != 1 || k >= w[NR]) return;
  int src = w[SL + hot * DC], dest = w[RP + k], sr = src / DS, dr = dest / DS, victim = m[dest], total = q[src];
  if (total == 0 || sr == dr || victim == hot) return;
  int t[DR], b[DR], bm = 0, tm = 0;
  for (int r = 0; r < DR; r++) {
    t[r] = w[TT + r];
    b[r] = w[BB + r];
    bm = max(bm, b[r]);
    tm = max(tm, t[r]);
  }
  t[sr] -= total;
  b[sr] -= dc_cb(total);
  int n = w[NS + victim], vt = w[LT + victim], ix = 0;
  for (int j = 0; j < n; j++) {
    int p = w[SL + victim * DC + j], r = p / DS;
    t[r] -= q[p];
    b[r] -= dc_cb(q[p]);
  }
  for (int j = 0; j < n; j++) {
    int p = w[SL + victim * DC + j];
    if (p == dest) continue;
    int v = vt / (n - 1) + (ix < vt % (n - 1));
    ix++;
    t[p / DS] += v;
    b[p / DS] += dc_cb(v);
  }
  int baseB = 0, baseT = 0;
  for (int r = 0; r < DR; r++) {
    baseB = max(baseB, b[r]);
    baseT = max(baseT, t[r]);
  }
  if (baseB > bm - 1 || baseT > tm || dc_cap(sr, bm - 1, tm, b, t) + dc_cap(dr, bm - 1, tm, b, t) < total) return;
  int lo = baseB, hi = bm - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (dc_cap(sr, mid, tm, b, t) + dc_cap(dr, mid, tm, b, t) >= total)
      hi = mid;
    else
      lo = mid + 1;
  }
  int B = lo;
  lo = min(t[sr], t[dr]);
  hi = tm;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (dc_cap(sr, B, mid, b, t) + dc_cap(dr, B, mid, b, t) >= total)
      hi = mid;
    else
      lo = mid + 1;
  }
  int ranks[2] = {min(sr, dr), max(sr, dr)}, v[2] = {dc_cap(ranks[0], B, lo, b, t), dc_cap(ranks[1], B, lo, b, t)};
  int extra = v[0] + v[1] - total, first = t[ranks[0]] + v[0] > t[ranks[1]] + v[1] ? 0 : 1;
  for (int j = 0; j < 2; j++) {
    int i = j == 0 ? first : 1 - first, take = min(extra, v[i]);
    v[i] -= take;
    extra -= take;
  }
  int qs = v[sr == ranks[0] ? 0 : 1], qd = v[dr == ranks[0] ? 0 : 1];
  if (qd == 0 || extra) return;
  for (int j = 0; j < 2; j++) {
    t[ranks[j]] += v[j];
    b[ranks[j]] += dc_cb(v[j]);
  }
  int nb = 0, nt = 0, sumB = 0;
  int64_t sumT2 = 0;
  for (int r = 0; r < DR; r++) {
    nb = max(nb, b[r]);
    nt = max(nt, t[r]);
    sumB += b[r];
    sumT2 += (int64_t)t[r] * t[r];
  }
  key[0] = nb;
  key[1] = nt;
  key[2] = sumB;
  key[3] = sumT2;
  key[4] = dest;
  key[5] = hot;
  key[6] = qs;
  key[7] = qd;
}
__device__ bool dc_better(const int64_t* keys, int a, int b) {
  if (a < 0) return false;
  if (b < 0) return true;
  if (keys[a * 8] == INT64_MAX) return false;
  if (keys[b * 8] == INT64_MAX) return true;
  for (int j = 0; j < 6; j++) {
    int64_t x = keys[a * 8 + j], y = keys[b * 8 + j];
    if (x != y) return x < y;
  }
  return a < b;
}
__global__ void
dc_select_kernel(const int64_t* m, const int* q, const int* w, const int64_t* keys, int* plan, int64_t* om, int* oq) {
  __shared__ int best[256];
  int tid = threadIdx.x, win = -1;
  for (int i = tid; i < DN; i += 256)
    if (dc_better(keys, i, win)) win = i;
  best[tid] = win;
  __syncthreads();
  for (int s = 128; s; s /= 2) {
    if (tid < s && dc_better(keys, best[tid + s], best[tid])) best[tid] = best[tid + s];
    __syncthreads();
  }
  win = best[0];
  bool change = win >= 0 && keys[win * 8] != INT64_MAX;
  int hot = change ? keys[win * 8 + 5] : -1, dest = change ? keys[win * 8 + 4] : -1;
  int src = change ? w[SL + hot * DC] : -1, victim = change ? m[dest] : -1;
  if (tid == 0) {
    int bm = 0, tm = 0;
    for (int r = 0; r < DR; r++) {
      bm = max(bm, w[BB + r]);
      tm = max(tm, w[TT + r]);
    }
    plan[0] = change;
    plan[1] = hot;
    plan[2] = victim;
    plan[3] = src;
    plan[4] = dest;
    plan[5] = bm;
    plan[6] = change ? keys[win * 8] : bm;
    plan[7] = tm;
    plan[8] = change ? keys[win * 8 + 1] : tm;
  }
  for (int p = tid; p < DP; p += 256) {
    int v = q[p];
    int64_t logical = m[p];
    if (change) {
      if (p == src)
        v = keys[win * 8 + 6];
      else if (p == dest) {
        v = keys[win * 8 + 7];
        logical = hot;
      } else if (logical == victim) {
        int ix = 0, n = w[NS + victim] - 1;
        for (int j = 0; j < w[NS + victim]; j++) {
          int s = w[SL + victim * DC + j];
          if (s == p) break;
          if (s != dest) ix++;
        }
        v = w[LT + victim] / n + (ix < w[LT + victim] % n);
      }
    }
    om[p] = logical;
    oq[p] = v;
  }
}
void dc_plan(DCV m, DCV q, DCV work, DCV keys, DCV plan, DCV om, DCV oq) {
  using namespace sglang::host;
  SymbolicDevice d;
  TensorMatcher({DP}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(m).verify(om);
  TensorMatcher({DP}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(q).verify(oq);
  TensorMatcher({DWORK}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(work);
  TensorMatcher({DN, 8}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(keys);
  TensorMatcher({9}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(plan);
  LaunchKernel(1, 256, d.unwrap())(
      dc_prepare_kernel, (const int64_t*)m.data_ptr(), (const int*)q.data_ptr(), (int*)work.data_ptr());
  LaunchKernel((DN + 127) / 128, 128, d.unwrap())(
      dc_candidates_kernel,
      (const int64_t*)m.data_ptr(),
      (const int*)q.data_ptr(),
      (const int*)work.data_ptr(),
      (int64_t*)keys.data_ptr());
  LaunchKernel(1, 256, d.unwrap())(
      dc_select_kernel,
      (const int64_t*)m.data_ptr(),
      (const int*)q.data_ptr(),
      (const int*)work.data_ptr(),
      (const int64_t*)keys.data_ptr(),
      (int*)plan.data_ptr(),
      (int64_t*)om.data_ptr(),
      (int*)oq.data_ptr());
}

// Shape-stable device metadata. Padding pairs (0,0) are inert in BlockLPLB.
__global__ void dc_metadata_kernel(
    const int64_t* m, const int* w, int64_t* physical, int64_t* valid, int* phy, int* pairs, int* lut, int rank) {
  for (int e = threadIdx.x; e < DL; e += blockDim.x) {
    int n = w[NS + e], local = -1;
    lut[e] = w[SL + e * DC];
    for (int j = 0; j < n; j++) {
      int p = w[SL + e * DC + j];
      if (p / DS == rank && local < 0) local = p;
      physical[p] = e;
      phy[p] = e;
      phy[DP + p] = w[SL + e * DC + (j + 1) % n];
    }
    valid[e] = local >= 0 ? 1 : n;
  }
  if (threadIdx.x == 0) {
    int nr = w[NR], k = 0;
    for (int i = 0; i < nr; i++)
      for (int j = i + 1; j < nr; j++) {
        int a = w[RP + i], b = w[RP + j];
        if (m[a] == m[b] && a / DS != b / DS) {
          pairs[k * 2] = a;
          pairs[k * 2 + 1] = b;
          k++;
        }
      }
    for (; k < DPAIRS; k++) {
      pairs[k * 2] = 0;
      pairs[k * 2 + 1] = 0;
    }
  }
}
__global__ void dc_metadata_dense_kernel(const int* w, int64_t* logical, int rank, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= DL * width) return;
  int e = i / width, j = i % width, n = w[NS + e], local = -1;
  for (int k = 0; k < n; k++) {
    int p = w[SL + e * DC + k];
    if (p / DS == rank) {
      local = p;
      break;
    }
  }
  logical[i] = (local >= 0 ? (j == 0 ? local : -1) : (j < n ? w[SL + e * DC + j] : -1));
}
void dc_metadata(DCV m, DCV physical, DCV logical, DCV valid, DCV phy, DCV pairs, DCV lut, DCV work, int rank) {
  using namespace sglang::host;
  SymbolicDevice d;
  SymbolicSize W{"width"};
  TensorMatcher({DP}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(m).verify(physical);
  TensorMatcher({DL, W}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(logical);
  TensorMatcher({DL}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(valid);
  TensorMatcher({2 * DP}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(phy);
  TensorMatcher({DPAIRS, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(pairs);
  TensorMatcher({DL}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(lut);
  TensorMatcher({DWORK}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(work);
  LaunchKernel(1, 256, d.unwrap())(
      dc_prepare_kernel, (const int64_t*)m.data_ptr(), (const int*)nullptr, (int*)work.data_ptr());
  LaunchKernel(1, 256, d.unwrap())(
      dc_metadata_kernel,
      (const int64_t*)m.data_ptr(),
      (const int*)work.data_ptr(),
      (int64_t*)physical.data_ptr(),
      (int64_t*)valid.data_ptr(),
      (int*)phy.data_ptr(),
      (int*)pairs.data_ptr(),
      (int*)lut.data_ptr(),
      rank);
  LaunchKernel((DL * W.unwrap() + 255) / 256, 256, d.unwrap())(
      dc_metadata_dense_kernel, (const int*)work.data_ptr(), (int64_t*)logical.data_ptr(), rank, (int)W.unwrap());
}

// The destination reads ORIGINAL source storage into staging. No active write
// is permitted until the caller's all-rank device collective completes.
__global__ void
dc_copy_kernel(const int* plan, const int64_t* addresses, int tensor_count, int rank, int phase, int* ok) {
  if (!plan[0] || plan[4] / DS != rank) return;
  int t = blockIdx.y, sr = plan[3] / DS, srcslot = plan[3] % DS, dstslot = plan[4] % DS;
  const int64_t* a = addresses + t * (DR + 3);
  const uint4* src = (const uint4*)(phase == 0 ? a[sr] + a[DR + 1] * srcslot : a[DR]);
  uint4* dst = (uint4*)(phase == 0 ? a[DR] : a[rank] + a[DR + 1] * dstslot);
  int64_t n = a[DR + 2] / 16;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    uint4 v = src[i];
    if (phase < 2)
      dst[i] = v;
    else {
      uint4 x = dst[i];
      if (x.x != v.x || x.y != v.y || x.z != v.z || x.w != v.w) atomicExch(ok, 0);
    }
  }
}
void dc_copy(DCV plan, DCV addresses, DCV ok, int rank, int phase, int blocks) {
  using namespace sglang::host;
  SymbolicDevice d;
  SymbolicSize NT{"tensors"};
  TensorMatcher({9}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(plan);
  TensorMatcher({NT, DR + 3}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(addresses);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(ok);
  LaunchKernel(dim3(blocks, NT.unwrap()), 256, d.unwrap())(
      dc_copy_kernel,
      (const int*)plan.data_ptr(),
      (const int64_t*)addresses.data_ptr(),
      (int)NT.unwrap(),
      rank,
      phase,
      (int*)ok.data_ptr());
}

// Canonical source counts and routing are derived entirely on device.
__global__ void
dc_counts_kernel(const int64_t* newmap, const int* source, const int* lut, const int* oldwork, int* counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= DR * DP) return;
  int rank = i / DP, p = i % DP, v = 0;
  int e = newmap[p];
  if (lut[e] == p)
    for (int j = 0; j < oldwork[NS + e]; j++)
      v += source[rank * DP + oldwork[SL + e * DC + j]];
  counts[i] = v;
}
__global__ void dc_route_kernel(
    const int* input,
    int* out,
    int n,
    const int* plan,
    const int64_t* oldmap,
    const int* lut,
    const int* phy,
    const int* counts,
    const int* quota,
    int* cursor,
    int rank) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int old = input[i];
  if (old < 0 || !plan[0]) {
    out[i] = old;
    return;
  }
  int logical = oldmap[old], canonical = lut[logical], p = canonical, prefix = 0;
  bool changed = false;
  do {
    int total = 0;
    for (int r = 0; r < DR; r++) {
      int v = counts[r * DP + p];
      total += v;
      if (r < rank) prefix += v;
    }
    changed |= total != quota[p];
    p = phy[DP + p];
  } while (p != canonical);
  if (!changed) {
    out[i] = canonical;
    return;
  }
  int ordinal = prefix + atomicAdd(cursor + logical, 1), cum = 0, chosen = canonical;
  p = canonical;
  do {
    cum += quota[p];
    if (ordinal < cum) {
      chosen = p;
      break;
    }
    p = phy[DP + p];
  } while (p != canonical);
  out[i] = chosen;
}
void dc_counts(DCV newmap, DCV source, DCV lut, DCV oldwork, DCV counts) {
  using namespace sglang::host;
  SymbolicDevice d;
  TensorMatcher({DP}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(newmap);
  TensorMatcher({DR, DP}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(source).verify(counts);
  TensorMatcher({DL}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(lut);
  TensorMatcher({DWORK}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(oldwork);
  LaunchKernel((DR * DP + 255) / 256, 256, d.unwrap())(
      dc_counts_kernel,
      (const int64_t*)newmap.data_ptr(),
      (const int*)source.data_ptr(),
      (const int*)lut.data_ptr(),
      (const int*)oldwork.data_ptr(),
      (int*)counts.data_ptr());
}
void dc_route(DCV input, DCV out, DCV plan, DCV oldmap, DCV lut, DCV phy, DCV counts, DCV quota, DCV cursor, int rank) {
  using namespace sglang::host;
  SymbolicDevice d;
  SymbolicSize N{"tokens_topk"};
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(input).verify(out);
  TensorMatcher({9}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(plan);
  TensorMatcher({DP}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(oldmap);
  TensorMatcher({DL}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(lut).verify(cursor);
  TensorMatcher({2 * DP}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(phy);
  TensorMatcher({DR, DP}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(counts);
  TensorMatcher({DP}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(quota);
  int n = N.unwrap();
  if (n)
    LaunchKernel((n + 255) / 256, 256, d.unwrap())(
        dc_route_kernel,
        (const int*)input.data_ptr(),
        (int*)out.data_ptr(),
        n,
        (const int*)plan.data_ptr(),
        (const int64_t*)oldmap.data_ptr(),
        (const int*)lut.data_ptr(),
        (const int*)phy.data_ptr(),
        (const int*)counts.data_ptr(),
        (const int*)quota.data_ptr(),
        (int*)cursor.data_ptr(),
        rank);
}
}  // namespace
namespace {
__global__ void
dc_snapshot_kernel(const int* info, const int* before, const int64_t* mapping, const int* quota, int* packet) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 9) packet[i] = info[i];
  if (i < DP) {
    packet[9 + i] = before[i];
    packet[9 + DP + i] = mapping[i];
    packet[9 + 2 * DP + i] = quota[i];
  }
}
void dc_snapshot(DCV info, DCV before, DCV mapping, DCV quota, DCV packet) {
  using namespace sglang::host;
  SymbolicDevice d;
  TensorMatcher({9}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(info);
  TensorMatcher({DP}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(before).verify(quota);
  TensorMatcher({DP}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(mapping);
  TensorMatcher({9 + 3 * DP}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(packet);
  LaunchKernel((DP + 255) / 256, 256, d.unwrap())(
      dc_snapshot_kernel,
      (const int*)info.data_ptr(),
      (const int*)before.data_ptr(),
      (const int64_t*)mapping.data_ptr(),
      (const int*)quota.data_ptr(),
      (int*)packet.data_ptr());
}
}  // namespace
