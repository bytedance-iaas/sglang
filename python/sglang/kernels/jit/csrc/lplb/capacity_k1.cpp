// Exact singleton K1 capacity search. Integer arithmetic; same lexicographic
// objective and water-filling tie order as capacity_planner.choose(...,False).
#include "wan22_shape.cuh"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
#ifndef WAN22_BALANCE_BLOCK_SIZE
#define WAN22_BALANCE_BLOCK_SIZE 128
#endif
static_assert(WAN22_BALANCE_BLOCK_SIZE == 64 || WAN22_BALANCE_BLOCK_SIZE == 128);
using I = int64_t;
extern "C" int plan_k1(const I* m, const I* q, I* outm, I* outq, I* info) {
  constexpr int P = DP, L = DL, R = DR, S = DS, M = WAN22_BALANCE_BLOCK_SIZE;
  std::array<std::vector<int>, L> slots;
  std::array<I, R> t0{}, b0{};
  for (int p = 0; p < P; p++) {
    if (m[p] < 0 || m[p] >= L || q[p] < 0) return -1;
    slots[m[p]].push_back(p);
    t0[p / S] += q[p];
    b0[p / S] += (q[p] + M - 1) / M;
  }
  for (int e = 0; e < L; e++)
    if (slots[e].empty()) return -2;
  I bm = *std::max_element(b0.begin(), b0.end()), tm = *std::max_element(t0.begin(), t0.end());
  bool found = false;
  std::array<I, 6> best{};
  for (int hot = 0; hot < L; hot++) {
    if (slots[hot].size() != 1) continue;
    int src = slots[hot][0], sr = src / S;
    I total = q[src];
    if (total == 0) continue;
    for (int dest = 0; dest < P; dest++) {
      int victim = m[dest], dr = dest / S;
      if (slots[victim].size() < 2 || victim == hot || dr == sr) continue;
      auto t = t0, b = b0;
      t[sr] -= total;
      b[sr] -= (total + M - 1) / M;
      I vtotal = 0;
      for (int p : slots[victim]) {
        vtotal += q[p];
        t[p / S] -= q[p];
        b[p / S] -= (q[p] + M - 1) / M;
      }
      std::vector<std::pair<int, I>> vq;
      int count = slots[victim].size() - 1, ix = 0;
      for (int p : slots[victim])
        if (p != dest) {
          I n = vtotal / count + (ix < vtotal % count);
          ix++;
          vq.push_back({p, n});
          t[p / S] += n;
          b[p / S] += (n + M - 1) / M;
        }
      I baseB = *std::max_element(b.begin(), b.end()), baseT = *std::max_element(t.begin(), t.end());
      if (baseB > bm - 1 || baseT > tm) continue;
      int ranks[2] = {std::min(sr, dr), std::max(sr, dr)};
      auto cap = [&](int r, I B, I T) { return std::min((B - b[r]) * M, std::max(I(0), T - t[r])); };
      if (cap(sr, bm - 1, tm) + cap(dr, bm - 1, tm) < total) continue;
      I lo = baseB, hi = bm - 1;
      while (lo < hi) {
        I mid = (lo + hi) / 2;
        if (cap(sr, mid, tm) + cap(dr, mid, tm) >= total)
          hi = mid;
        else
          lo = mid + 1;
      }
      I B = lo;
      lo = std::min(t[sr], t[dr]);
      hi = tm;
      while (lo < hi) {
        I mid = (lo + hi) / 2;
        if (cap(sr, B, mid) + cap(dr, B, mid) >= total)
          hi = mid;
        else
          lo = mid + 1;
      }
      I vals[2] = {cap(ranks[0], B, lo), cap(ranks[1], B, lo)}, extra = vals[0] + vals[1] - total;
      int first = (t[ranks[0]] + vals[0] > t[ranks[1]] + vals[1]) ? 0 : 1;
      for (int i : {first, 1 - first}) {
        I n = std::min(extra, vals[i]);
        vals[i] -= n;
        extra -= n;
      }
      if (extra) return -3;
      I hsrc = vals[sr == ranks[0] ? 0 : 1], hdst = vals[dr == ranks[0] ? 0 : 1];
      if (!hdst) continue;
      for (int i = 0; i < 2; i++) {
        int r = ranks[i];
        t[r] += vals[i];
        b[r] += (vals[i] + M - 1) / M;
      }
      I sumB = 0, sumT2 = 0;
      for (int r = 0; r < R; r++) {
        sumB += b[r];
        sumT2 += t[r] * t[r];
      }
      std::array<I, 6> key = {
          *std::max_element(b.begin(), b.end()), *std::max_element(t.begin(), t.end()), sumB, sumT2, dest, hot};
      if (found && key >= best) continue;
      found = true;
      best = key;
      std::copy(m, m + P, outm);
      std::copy(q, q + P, outq);
      outm[dest] = hot;
      outq[src] = hsrc;
      outq[dest] = hdst;
      for (auto [p, n] : vq)
        outq[p] = n;
      info[0] = hot;
      info[1] = victim;
      info[2] = src;
      info[3] = dest;
      info[4] = bm;
      info[5] = key[0];
      info[6] = tm;
      info[7] = key[1];
    }
  }
  return found ? 1 : 0;
}
