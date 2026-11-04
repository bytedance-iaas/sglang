#pragma once
#include "device_control.cuh"
namespace {
using MI = int64_t;
__device__ bool mm_less(const MI* a, const MI* b) {
  for (int k = 0; k < 6; k++) {
    if (a[k] != b[k]) return a[k] < b[k];
  }
  return false;
}
__device__ void mm_stats(
    const int* old,
    const int* m,
    const int* q,
    const int* origin,
    int* t,
    int* b,
    int* sent,
    int* recv,
    int* mask,
    int* hot,
    int* nhot,
    int* bm,
    int* nm,
    MI* b2) {
  int id = threadIdx.x;
  if (id < DR) {
    t[id] = 0;
    b[id] = 0;
    sent[id] = 0;
    recv[id] = 0;
  }
  for (int e = id; e < DL; e += blockDim.x)
    mask[e] = 0;
  __syncthreads();
  for (int p = id; p < DP; p += 256) {
    int r = p / DS;
    atomicAdd(t + r, q[p]);
    atomicAdd(b + r, dc_cb(q[p]));
    atomicOr(mask + m[p], 1u << r);
    if (m[p] != old[p]) {
      atomicAdd(sent + origin[p] / DS, 1);
      atomicAdd(recv + r, 1);
    }
  }
  __syncthreads();
  if (id == 0) {
    *bm = 0;
    *b2 = 0;
    *nm = 0;
    *nhot = 0;
    for (int r = 0; r < DR; r++) {
      *bm = max(*bm, b[r]);
      *b2 += (MI)b[r] * b[r];
    }
    for (int r = 0; r < DR; r++)
      *nm += b[r] == *bm;
    for (int p = 0; p < DP; p++)
      if (b[p / DS] == *bm && q[p] > 0) hot[(*nhot)++] = p;
  }
  __syncthreads();
}
__global__ void mm_improve_kernel(
    const int64_t* om,
    const int* oq,
    const int* baseline,
    int64_t* gm,
    int* gq,
    int* info,
    int* copies,
    int* counter,
    int* history,
    int budget) {
  __shared__ int old[DP], m[DP], q[DP], origin[DP], basem[DP], baseq[DP];
  __shared__ int t[DR], b[DR], sent[DR], recv[DR], mask[DL], hot[DP], nhot, bm, nm, baseB, baseT, nswap, plateau, loops,
      active, rollback;
  __shared__ MI b2, keys[256][6];
  __shared__ int win[256];
  int id = threadIdx.x;
  for (int p = id; p < DP; p += 256) {
    old[p] = om[p];
    m[p] = basem[p] = gm[p];
    q[p] = baseq[p] = gq[p];
    origin[p] = (baseline[0] && p == baseline[4]) ? baseline[3] : p;
  }
  for (int j = id; j < 2 * MCOPIES; j += blockDim.x)
    copies[j] = -1;
  if (id < 32) history[id] = -1;
  if (id == 0) {
    baseB = baseline[6];
    baseT = baseline[8];
    nswap = 0;
    plateau = 0;
    loops = 0;
    active = 1;
  }
  __syncthreads();
  mm_stats(old, m, q, origin, t, b, sent, recv, mask, hot, &nhot, &bm, &nm, &b2);
  for (int step = 0; step < 16; step++) {
    if (id == 0) loops++;
    MI best[6] = {INT64_MAX, 0, 0, 0, 0, 0};
    for (int ix = id; ix < nhot * DP; ix += 256) {
      int p = hot[ix / DP], d = ix % DP, sr = p / DS, dr = d / DS;
      if (sr == dr || b[dr] >= bm || m[p] == m[d] || (mask[m[p]] & (1u << dr)) || (mask[m[d]] & (1u << sr))) continue;
      int db = dc_cb(q[p]) - dc_cb(q[d]), dt = q[p] - q[d];
      if (db <= 0 || dt <= 0) continue;
      int op = m[p] != old[p], od = m[d] != old[d], np = m[d] != old[p], nd = m[p] != old[d], ps = origin[p] / DS,
          ds = origin[d] / DS;
      int nb = 0, nt = 0, nn = 0;
      MI sb2 = 0, st2 = 0;
      bool bad = false;
      for (int r = 0; r < DR; r++) {
        int so = sent[r] - (r == ps) * op - (r == ds) * od + (r == ds) * np + (r == ps) * nd;
        int ri = recv[r] - (r == sr) * op - (r == dr) * od + (r == sr) * np + (r == dr) * nd;
        int bv = b[r] - (r == sr) * db + (r == dr) * db, tv = t[r] - (r == sr) * dt + (r == dr) * dt;
        if (so > budget || ri > budget || so < 0 || ri < 0 || tv > baseT || bv > bm) {
          bad = true;
          break;
        }
        if (bv > nb) {
          nb = bv;
          nn = 1;
        } else if (bv == nb)
          nn++;
        nt = max(nt, tv);
        sb2 += (MI)bv * bv;
        st2 += (MI)tv * tv;
      }
      if (bad || !(nb < bm || (nb == bm && (nn < nm || (nn == nm && sb2 < b2))))) continue;
      MI key[6] = {nb, nn, sb2, nt, st2, p * DP + d};
      if (mm_less(key, best))
        for (int k = 0; k < 6; k++)
          best[k] = key[k];
    }
    for (int k = 0; k < 6; k++)
      keys[id][k] = best[k];
    win[id] = id;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
      if (id < stride && mm_less(keys[win[id + stride]], keys[win[id]])) win[id] = win[id + stride];
      __syncthreads();
    }
    if (id == 0) {
      MI* key = keys[win[0]];
      active = key[0] != INT64_MAX;
      if (active) {
        int p = key[5] / DP, d = key[5] % DP;
        if (key[0] == bm) plateau++;
        history[2 * nswap] = p;
        history[2 * nswap + 1] = d;
        nswap++;
        int z = m[p];
        m[p] = m[d];
        m[d] = z;
        z = q[p];
        q[p] = q[d];
        q[d] = z;
        z = origin[p];
        origin[p] = origin[d];
        origin[d] = z;
      }
    }
    __syncthreads();
    if (!active) break;
    mm_stats(old, m, q, origin, t, b, sent, recv, mask, hot, &nhot, &bm, &nm, &b2);
  }
  if (id == 0) rollback = bm >= baseB;
  __syncthreads();
  if (rollback) {
    for (int p = id; p < DP; p += 256) {
      m[p] = basem[p];
      q[p] = baseq[p];
      origin[p] = (baseline[0] && p == baseline[4]) ? baseline[3] : p;
    }
    if (id < 32) history[id] = -1;
    if (id == 0) nswap = 0;
    __syncthreads();
    mm_stats(old, m, q, origin, t, b, sent, recv, mask, hot, &nhot, &bm, &nm, &b2);
  }
  for (int p = id; p < DP; p += 256) {
    gm[p] = m[p];
    gq[p] = q[p];
  }
  if (id < DR) {
    counter[id] = sent[id];
    counter[DR + id] = recv[id];
  }
  if (id == 0) {
    int n = 0, tm = 0;
    for (int p = 0; p < DP; p++)
      if (m[p] != old[p]) {
        copies[2 * n] = origin[p];
        copies[2 * n + 1] = p;
        n++;
      }
    for (int r = 0; r < DR; r++)
      tm = max(tm, t[r]);
    int v[12] = {n, nswap, baseline[5], bm, baseline[7], tm, baseline[0], budget, baseB, baseT, plateau, loops};
    for (int k = 0; k < 12; k++)
      info[k] = v[k];
  }
}
void mm_improve(
    DCV oldm,
    DCV oldq,
    DCV baseline,
    DCV mapping,
    DCV quota,
    DCV info,
    DCV copies,
    DCV counter,
    DCV history,
    int budget) {
  using namespace sglang::host;
  SymbolicDevice d;
  TensorMatcher({DP}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(oldm).verify(mapping);
  TensorMatcher({DP}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(oldq).verify(quota);
  TensorMatcher({9}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(baseline);
  TensorMatcher({12}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(info);
  TensorMatcher({MCOPIES, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(copies);
  TensorMatcher({2 * DR}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(counter);
  TensorMatcher({16, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(history);
  LaunchKernel(1, 256, d.unwrap())(
      mm_improve_kernel,
      (const int64_t*)oldm.data_ptr(),
      (const int*)oldq.data_ptr(),
      (const int*)baseline.data_ptr(),
      (int64_t*)mapping.data_ptr(),
      (int*)quota.data_ptr(),
      (int*)info.data_ptr(),
      (int*)copies.data_ptr(),
      (int*)counter.data_ptr(),
      (int*)history.data_ptr(),
      budget);
}
__global__ void
mm_copy_kernel(const int* info, const int* copies, const int64_t* addresses, int rank, int phase, int* ok) {
  if (info[0] == 0) return;
  int ordinal = blockIdx.z, seen = 0, source = -1, dest = -1;
  for (int j = 0; j < info[0]; j++)
    if (copies[2 * j + 1] / DS == rank) {
      if (seen++ == ordinal) {
        source = copies[2 * j];
        dest = copies[2 * j + 1];
        break;
      }
    }
  if (dest < 0) return;
  const int64_t* a = addresses + blockIdx.y * (DR + 6);
  int sr = source / DS;
  const uint4* src = (const uint4*)(phase == 0 ? a[sr] + a[DR + 4] * (source % DS) : a[DR + ordinal]);
  uint4* dst = (uint4*)(phase == 0 ? a[DR + ordinal] : a[rank] + a[DR + 4] * (dest % DS));
  int64_t n = a[DR + 5] / 16;
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
void mm_copy(DCV info, DCV copies, DCV addresses, DCV ok, int rank, int phase, int blocks, int budget) {
  using namespace sglang::host;
  SymbolicDevice d;
  SymbolicSize NT{"tensors"};
  TensorMatcher({12}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(info);
  TensorMatcher({MCOPIES, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(copies);
  TensorMatcher({NT, DR + 6}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(addresses);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(ok);
  LaunchKernel(dim3(blocks, NT.unwrap(), budget), 256, d.unwrap())(
      mm_copy_kernel,
      (const int*)info.data_ptr(),
      (const int*)copies.data_ptr(),
      (const int64_t*)addresses.data_ptr(),
      rank,
      phase,
      (int*)ok.data_ptr());
}
__global__ void mm_snapshot_kernel(
    const int* info,
    const int* copies,
    const int* history,
    const int* before,
    const int64_t* m,
    const int* q,
    const int* counts,
    int* packet) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 12) packet[i] = info[i];
  if (i < 2 * MCOPIES) packet[12 + i] = copies[i];
  if (i < 32) packet[12 + 2 * MCOPIES + i] = history[i];
  if (i < DP) {
    packet[MSNAP + i] = before[i];
    packet[MSNAP + DP + i] = m[i];
    packet[MSNAP + 2 * DP + i] = q[i];
  }
  if (i < 2 * DR) packet[MSNAP + 3 * DP + i] = counts[i];
}
void mm_snapshot(DCV info, DCV copies, DCV history, DCV before, DCV mapping, DCV quota, DCV counts, DCV packet) {
  using namespace sglang::host;
  SymbolicDevice d;
  TensorMatcher({12}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(info);
  TensorMatcher({MCOPIES, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(copies);
  TensorMatcher({16, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(history);
  TensorMatcher({DP}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(before).verify(quota);
  TensorMatcher({DP}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(mapping);
  TensorMatcher({2 * DR}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(counts);
  TensorMatcher({MSNAP + 3 * DP + 2 * DR}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(packet);
  LaunchKernel((max(DP, 2 * MCOPIES) + 255) / 256, 256, d.unwrap())(
      mm_snapshot_kernel,
      (const int*)info.data_ptr(),
      (const int*)copies.data_ptr(),
      (const int*)history.data_ptr(),
      (const int*)before.data_ptr(),
      (const int64_t*)mapping.data_ptr(),
      (const int*)quota.data_ptr(),
      (const int*)counts.data_ptr(),
      (int*)packet.data_ptr());
}
}  // namespace

namespace {
// Parallel candidate evaluation; persistent state remains on device.
__device__ void mm_stats2(int* w) {
  int id = threadIdx.x;
  if (id < DR) {
    w[WT + id] = 0;
    w[WB + id] = 0;
    w[WS + id] = 0;
    w[WR + id] = 0;
  }
  for (int e = id; e < DL; e += blockDim.x)
    w[WMASK + e] = 0;
  __syncthreads();
  for (int p = id; p < DP; p += 256) {
    int r = p / DS;
    atomicAdd(w + WT + r, w[WQ + p]);
    atomicAdd(w + WB + r, dc_cb(w[WQ + p]));
    atomicOr(w + WMASK + w[WM + p], 1u << r);
    if (w[WM + p] != w[WOLD + p]) {
      atomicAdd(w + WS + w[WO + p] / DS, 1);
      atomicAdd(w + WR + r, 1);
    }
  }
  __syncthreads();
  if (id == 0) {
    int bm = 0, b2 = 0, nm = 0, nh = 0;
    for (int r = 0; r < DR; r++) {
      bm = max(bm, w[WB + r]);
      b2 += w[WB + r] * w[WB + r];
    }
    for (int r = 0; r < DR; r++)
      nm += w[WB + r] == bm;
    w[WMAX] = bm;
    w[WB2] = b2;
    w[WNMAX] = nm;
    w[WNH] = 0;
  }
  __syncthreads();
  for (int p = id; p < DP; p += 256)
    if (w[WB + p / DS] == w[WMAX] && w[WQ + p] > 0) {
      int n = atomicAdd(w + WNH, 1);
      w[WHOT + n] = p;
    }
  __syncthreads();
}
__global__ void mm_init2(const int64_t* oldm, const int* bi, const int64_t* m, const int* q, int* w, int* history) {
  int id = threadIdx.x;
  for (int p = id; p < DP; p += 256) {
    w[WOLD + p] = oldm[p];
    w[WM + p] = w[WBM + p] = m[p];
    w[WQ + p] = w[WBQ + p] = q[p];
    w[WO + p] = (bi[0] && p == bi[4]) ? bi[3] : p;
  }
  if (id < 32) history[id] = -1;
  if (id == 0) {
    w[WBASEB] = bi[6];
    w[WBASET] = bi[8];
    w[WNS] = 0;
    w[WPL] = 0;
    w[WL] = 0;
    w[WA] = 1;
  }
  __syncthreads();
  mm_stats2(w);
}
__global__ void mm_candidates2(const int* w, MI* output, int budget) {
  if (!w[WA]) return;
  __shared__ int m[DP], q[DP], origin[DP], old[DP], mask[DL], hot[DP], b[DR], t[DR], sent[DR], recv[DR];
  __shared__ MI keys[256][6];
  __shared__ int win[256];
  int id = threadIdx.x;
  for (int p = id; p < DP; p += 256) {
    m[p] = w[WM + p];
    q[p] = w[WQ + p];
    origin[p] = w[WO + p];
    old[p] = w[WOLD + p];
    hot[p] = w[WHOT + p];
  }
  for (int e = id; e < DL; e += blockDim.x)
    mask[e] = w[WMASK + e];
  if (id < DR) {
    b[id] = w[WB + id];
    t[id] = w[WT + id];
    sent[id] = w[WS + id];
    recv[id] = w[WR + id];
  }
  __syncthreads();
  int bm = w[WMAX], nm = w[WNMAX], nh = w[WNH], cap = w[WBASET];
  MI b2 = w[WB2], best[6] = {INT64_MAX, 0, 0, 0, 0, 0};
  for (int ix = blockIdx.x * 256 + id; ix < nh * DP; ix += gridDim.x * 256) {
    int p = hot[ix / DP], d = ix % DP, sr = p / DS, dr = d / DS;
    if (sr == dr || b[dr] >= bm || m[p] == m[d] || (mask[m[p]] & (1u << dr)) || (mask[m[d]] & (1u << sr))) continue;
    int db = dc_cb(q[p]) - dc_cb(q[d]), dt = q[p] - q[d];
    if (db <= 0 || dt <= 0) continue;
    int op = m[p] != old[p], od = m[d] != old[d], np = m[d] != old[p], nd = m[p] != old[d], ps = origin[p] / DS,
        ds = origin[d] / DS;
    int nb = 0, nt = 0, nn = 0;
    MI sb2 = 0, st2 = 0;
    bool bad = false;
    for (int r = 0; r < DR; r++) {
      int so = sent[r] - (r == ps) * op - (r == ds) * od + (r == ds) * np + (r == ps) * nd;
      int ri = recv[r] - (r == sr) * op - (r == dr) * od + (r == sr) * np + (r == dr) * nd;
      int bv = b[r] - (r == sr) * db + (r == dr) * db, tv = t[r] - (r == sr) * dt + (r == dr) * dt;
      if (so > budget || ri > budget || so < 0 || ri < 0 || tv > cap || bv > bm) {
        bad = true;
        break;
      }
      if (bv > nb) {
        nb = bv;
        nn = 1;
      } else if (bv == nb)
        nn++;
      nt = max(nt, tv);
      sb2 += (MI)bv * bv;
      st2 += (MI)tv * tv;
    }
    if (bad || !(nb < bm || (nb == bm && (nn < nm || (nn == nm && sb2 < b2))))) continue;
    MI key[6] = {nb, nn, sb2, nt, st2, p * DP + d};
    if (mm_less(key, best))
      for (int k = 0; k < 6; k++)
        best[k] = key[k];
  }
  for (int k = 0; k < 6; k++)
    keys[id][k] = best[k];
  win[id] = id;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (id < stride && mm_less(keys[win[id + stride]], keys[win[id]])) win[id] = win[id + stride];
    __syncthreads();
  }
  if (id < 6) output[blockIdx.x * 6 + id] = keys[win[0]][id];
}
__global__ void mm_select2(int* w, const MI* keys, int* history) {
  if (!w[WA]) return;
  int id = threadIdx.x;
  __shared__ MI kk[64][6];
  __shared__ int win[64];
  for (int i = id; i < 384; i += 256)
    kk[i / 6][i % 6] = keys[i];
  if (id < 64) win[id] = id;
  __syncthreads();
  for (int stride = 32; stride; stride /= 2) {
    if (id < stride && mm_less(kk[win[id + stride]], kk[win[id]])) win[id] = win[id + stride];
    __syncthreads();
  }
  if (id == 0) {
    w[WL]++;
    const MI* key = kk[win[0]];
    if (key[0] == INT64_MAX)
      w[WA] = 0;
    else {
      int p = key[5] / DP, d = key[5] % DP;
      if (key[0] == w[WMAX]) w[WPL]++;
      history[2 * w[WNS]] = p;
      history[2 * w[WNS] + 1] = d;
      w[WNS]++;
      int z = w[WM + p];
      w[WM + p] = w[WM + d];
      w[WM + d] = z;
      z = w[WQ + p];
      w[WQ + p] = w[WQ + d];
      w[WQ + d] = z;
      z = w[WO + p];
      w[WO + p] = w[WO + d];
      w[WO + d] = z;
    }
  }
  __syncthreads();
  if (w[WA]) mm_stats2(w);
}
__global__ void
mm_finish2(const int* bi, int* w, int64_t* m, int* q, int* info, int* copies, int* counts, int* history, int budget) {
  int id = threadIdx.x;
  __shared__ int rollback;
  if (id == 0) rollback = w[WMAX] >= w[WBASEB];
  __syncthreads();
  if (rollback) {
    for (int p = id; p < DP; p += 256) {
      w[WM + p] = w[WBM + p];
      w[WQ + p] = w[WBQ + p];
      w[WO + p] = (bi[0] && p == bi[4]) ? bi[3] : p;
    }
    if (id < 32) history[id] = -1;
    if (id == 0) w[WNS] = 0;
    __syncthreads();
    mm_stats2(w);
  }
  for (int p = id; p < DP; p += 256) {
    m[p] = w[WM + p];
    q[p] = w[WQ + p];
  }
  if (id < DR) {
    counts[id] = w[WS + id];
    counts[DR + id] = w[WR + id];
  }
  for (int j = id; j < 2 * MCOPIES; j += blockDim.x)
    copies[j] = -1;
  __syncthreads();
  __shared__ int changed[DP], offset[DP], total;
  for (int p = id; p < DP; p += 256)
    changed[p] = w[WM + p] != w[WOLD + p];
  __syncthreads();
  if (id == 0) {
    int n = 0;
    for (int p = 0; p < DP; p++) {
      offset[p] = n;
      n += changed[p];
    }
    total = n;
  }
  __syncthreads();
  for (int p = id; p < DP; p += 256)
    if (changed[p]) {
      copies[2 * offset[p]] = w[WO + p];
      copies[2 * offset[p] + 1] = p;
    }
  if (id == 0) {
    int tm = 0;
    for (int r = 0; r < DR; r++)
      tm = max(tm, w[WT + r]);
    int v[12] = {total, w[WNS], bi[5], w[WMAX], bi[7], tm, bi[0], budget, w[WBASEB], w[WBASET], w[WPL], w[WL]};
    for (int k = 0; k < 12; k++)
      info[k] = v[k];
  }
}
void mm_parallel(
    DCV oldm,
    DCV oldq,
    DCV baseline,
    DCV mapping,
    DCV quota,
    DCV info,
    DCV copies,
    DCV counter,
    DCV history,
    DCV workspace,
    DCV keys,
    int budget) {
  using namespace sglang::host;
  SymbolicDevice d;
  TensorMatcher({DP}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(oldm).verify(mapping);
  TensorMatcher({DP}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(oldq).verify(quota);
  TensorMatcher({9}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(baseline);
  TensorMatcher({12}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(info);
  TensorMatcher({MCOPIES, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(copies);
  TensorMatcher({2 * DR}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(counter);
  TensorMatcher({16, 2}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(history);
  TensorMatcher({MWORK}).with_dtype<int32_t>().with_device<kDLCUDA>(d).verify(workspace);
  TensorMatcher({64, 6}).with_dtype<int64_t>().with_device<kDLCUDA>(d).verify(keys);
  const int* bi = (const int*)baseline.data_ptr();
  int* w = (int*)workspace.data_ptr();
  int* hist = (int*)history.data_ptr();
  MI* kk = (MI*)keys.data_ptr();
  LaunchKernel(1, 256, d.unwrap())(
      mm_init2,
      (const int64_t*)oldm.data_ptr(),
      bi,
      (const int64_t*)mapping.data_ptr(),
      (const int*)quota.data_ptr(),
      w,
      hist);
  for (int step = 0; step < 16; step++) {
    LaunchKernel(64, 256, d.unwrap())(mm_candidates2, w, kk, budget);
    LaunchKernel(1, 256, d.unwrap())(mm_select2, w, kk, hist);
  }
  LaunchKernel(1, 256, d.unwrap())(
      mm_finish2,
      bi,
      w,
      (int64_t*)mapping.data_ptr(),
      (int*)quota.data_ptr(),
      (int*)info.data_ptr(),
      (int*)copies.data_ptr(),
      (int*)counter.data_ptr(),
      hist,
      budget);
}

}  // namespace
