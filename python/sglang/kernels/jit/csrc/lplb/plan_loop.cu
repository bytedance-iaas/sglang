#include "wan22_shape.cuh"
#include <climits>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>
using MI = int64_t;
#ifndef WAN22_BALANCE_BLOCK_SIZE
#define WAN22_BALANCE_BLOCK_SIZE 128
#endif
static_assert(WAN22_BALANCE_BLOCK_SIZE == 64 || WAN22_BALANCE_BLOCK_SIZE == 128);
__device__ int dc_cb(int x) {
  return (x + WAN22_BALANCE_BLOCK_SIZE - 1) / WAN22_BALANCE_BLOCK_SIZE;
}

__device__ bool mm_less(const MI* a, const MI* b) {
  for (int k = 0; k < 6; k++) {
    if (a[k] != b[k]) return a[k] < b[k];
  }
  return false;
}

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
    atomicOr(w + WMASK + w[WM + p], 1 << r);
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
__global__ void mm_init2(
    const int64_t* oldm,
    const int* bi,
    const int64_t* m,
    const int* q,
    int* w,
    int* history,
    cudaGraphConditionalHandle loop) {
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
  if (id == 0) cudaGraphSetConditional(loop, 1);
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
    if (sr == dr || b[dr] >= bm || m[p] == m[d] || (mask[m[p]] & (1 << dr)) || (mask[m[d]] & (1 << sr))) continue;
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
__global__ void mm_select2(int* w, const MI* keys, int* history, cudaGraphConditionalHandle loop) {
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
  if (id == 0) cudaGraphSetConditional(loop, w[WA] && w[WL] < 16);
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

#define TRY(x)                           \
  do {                                   \
    auto e = (x);                        \
    if (e != cudaSuccess) return int(e); \
  } while (0)
struct State {
  cudaGraph_t graph = nullptr, body = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaGraphNode_t loopnode;
  cudaGraphConditionalHandle handle;
  const int64_t* oldm;
  const int* bi;
  int64_t* m;
  int *q, *info, *copies, *counter, *history, *w;
  MI* keys;
  int budget;
};
extern "C" int plan_create(void** out, void** p, int budget) {
  auto s = new State;
  *out = s;
  s->oldm = (int64_t*)p[0];
  s->bi = (int*)p[1];
  s->m = (int64_t*)p[2];
  s->q = (int*)p[3];
  s->info = (int*)p[4];
  s->copies = (int*)p[5];
  s->counter = (int*)p[6];
  s->history = (int*)p[7];
  s->w = (int*)p[8];
  s->keys = (MI*)p[9];
  s->budget = budget;
  TRY(cudaGraphCreate(&s->graph, 0));
  TRY(cudaGraphConditionalHandleCreate(&s->handle, s->graph, 0, cudaGraphCondAssignDefault));
  return 0;
}
extern "C" int plan_begin(void* opaque, int phase, void* stream) {
  auto s = (State*)opaque;
  TRY(cudaStreamBeginCaptureToGraph(
      (cudaStream_t)stream,
      phase == 1 ? s->body : s->graph,
      phase == 2 ? &s->loopnode : nullptr,
      nullptr,
      phase == 2 ? 1 : 0,
      cudaStreamCaptureModeThreadLocal));
  return 0;
}
extern "C" int plan_end(void* stream) {
  cudaGraph_t g;
  TRY(cudaStreamEndCapture((cudaStream_t)stream, &g));
  return 0;
}
extern "C" int plan_init(void* opaque, void* stream) {
  auto s = (State*)opaque;
  mm_init2<<<1, 256, 0, (cudaStream_t)stream>>>(s->oldm, s->bi, s->m, s->q, s->w, s->history, s->handle);
  return int(cudaGetLastError());
}
extern "C" int plan_body(void* opaque, void* stream) {
  auto s = (State*)opaque;
  mm_candidates2<<<64, 256, 0, (cudaStream_t)stream>>>(s->w, s->keys, s->budget);
  mm_select2<<<1, 256, 0, (cudaStream_t)stream>>>(s->w, s->keys, s->history, s->handle);
  return int(cudaGetLastError());
}
extern "C" int plan_finish(void* opaque, void* stream) {
  auto s = (State*)opaque;
  mm_finish2<<<1, 256, 0, (cudaStream_t)stream>>>(
      s->bi, s->w, s->m, s->q, s->info, s->copies, s->counter, s->history, s->budget);
  return int(cudaGetLastError());
}
extern "C" int plan_add_while(void* opaque) {
  auto s = (State*)opaque;
  size_t n = 0;
  TRY(cudaGraphGetNodes(s->graph, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n), leaves;
  TRY(cudaGraphGetNodes(s->graph, nodes.data(), &n));
  for (auto node : nodes) {
    size_t k = 0;
    TRY(cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &k));
    if (!k) leaves.push_back(node);
  }
  cudaGraphNodeParams p{};
  p.type = cudaGraphNodeTypeConditional;
  p.conditional.handle = s->handle;
  p.conditional.type = cudaGraphCondTypeWhile;
  p.conditional.size = 1;
  TRY(cudaGraphAddNode(&s->loopnode, s->graph, leaves.data(), nullptr, leaves.size(), &p));
  s->body = p.conditional.phGraph_out[0];
  return 0;
}
extern "C" int plan_instantiate(void* opaque) {
  auto s = (State*)opaque;
  TRY(cudaGraphInstantiate(&s->exec, s->graph, 0));
  return 0;
}
extern "C" int plan_launch(void* opaque, void* stream) {
  auto s = (State*)opaque;
  TRY(cudaGraphLaunch(s->exec, (cudaStream_t)stream));
  return 0;
}
extern "C" int plan_destroy(void* opaque) {
  auto s = (State*)opaque;
  if (!s) return 0;
  if (s->exec) TRY(cudaGraphExecDestroy(s->exec));
  if (s->graph) TRY(cudaGraphDestroy(s->graph));
  delete s;
  return 0;
}
