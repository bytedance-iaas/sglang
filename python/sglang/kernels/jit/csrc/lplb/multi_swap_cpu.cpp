// CPU oracle for a K1 baseline followed by joint swaps under per-rank budgets.
#include "capacity_k1.cpp"
#include <cstring>
extern "C" int
plan_multi(const I* oldm, const I* oldq, int budget, I* outm, I* outq, I* info, I* copies, I* counts, I* swaps) {
  if (budget < 1 || budget > 4) return -10;
  I km[DP], kq[DP], ki[8];
  int used = plan_k1(oldm, oldq, km, kq, ki);
  if (used < 0) return used;
  if (!used) {
    std::copy(oldm, oldm + DP, km);
    std::copy(oldq, oldq + DP, kq);
  }
  std::array<I, DP> m, q, origin;
  for (int p = 0; p < DP; p++) {
    m[p] = km[p];
    q[p] = kq[p];
    origin[p] = (used && p == ki[3]) ? ki[2] : p;
  }
  std::array<I, DR> t{}, bl{}, sent{}, received{};
  std::array<int, DL> masks{};
  auto recalc = [&]() {
    t.fill(0);
    bl.fill(0);
    sent.fill(0);
    received.fill(0);
    masks.fill(0);
    for (int p = 0; p < DP; p++) {
      int r = p / DS;
      t[r] += q[p];
      bl[r] += (q[p] + WAN22_BALANCE_BLOCK_SIZE - 1) / WAN22_BALANCE_BLOCK_SIZE;
      masks[m[p]] |= 1u << r;
      if (m[p] != oldm[p]) {
        sent[origin[p] / DS]++;
        received[r]++;
      }
    }
  };
  recalc();
  I baseB = *std::max_element(bl.begin(), bl.end()), baseT = *std::max_element(t.begin(), t.end());
  int nswap = 0, plateaus = 0, loops = 0;
  std::fill(swaps, swaps + 32, -1);
  for (int step = 0; step < 16; step++) {
    loops++;
    I bm = *std::max_element(bl.begin(), bl.end()), b2 = 0;
    int nm = 0;
    for (I x : bl) {
      nm += x == bm;
      b2 += x * x;
    }
    std::array<I, 6> best{};
    bool found = false;
    int sp = -1, dp = -1;
    for (int p = 0; p < DP; p++) {
      int sr = p / DS;
      if (bl[sr] != bm) continue;
      for (int d = 0; d < DP; d++) {
        int dr = d / DS;
        if (sr == dr || bl[dr] >= bm || m[p] == m[d] || (masks[m[p]] & (1u << dr)) || (masks[m[d]] & (1u << sr)))
          continue;
        I deltaB = (q[p] + WAN22_BALANCE_BLOCK_SIZE - 1) / WAN22_BALANCE_BLOCK_SIZE -
                   (q[d] + WAN22_BALANCE_BLOCK_SIZE - 1) / WAN22_BALANCE_BLOCK_SIZE,
          deltaT = q[p] - q[d];
        if (deltaB <= 0 || deltaT <= 0) continue;
        bool op = m[p] != oldm[p], od = m[d] != oldm[d], np = m[d] != oldm[p], nd = m[p] != oldm[d];
        int ps = origin[p] / DS, ds = origin[d] / DS;
        I nb = 0, nt = 0, sb2 = 0, st2 = 0;
        int nn = 0;
        bool bad = false;
        for (int r = 0; r < DR; r++) {
          I send = sent[r] - (r == ps) * op - (r == ds) * od + (r == ds) * np + (r == ps) * nd;
          I recv = received[r] - (r == sr) * op - (r == dr) * od + (r == sr) * np + (r == dr) * nd;
          I bv = bl[r] - (r == sr) * deltaB + (r == dr) * deltaB, tv = t[r] - (r == sr) * deltaT + (r == dr) * deltaT;
          if (send > budget || recv > budget || send < 0 || recv < 0 || tv > baseT || bv > bm) {
            bad = true;
            break;
          }
          if (bv > nb) {
            nb = bv;
            nn = 1;
          } else if (bv == nb)
            nn++;
          nt = std::max(nt, tv);
          sb2 += bv * bv;
          st2 += tv * tv;
        }
        if (bad || std::array<I, 3>{nb, nn, sb2} >= std::array<I, 3>{bm, nm, b2}) continue;
        std::array<I, 6> key = {nb, nn, sb2, nt, st2, p * DP + d};
        if (!found || key < best) {
          found = true;
          best = key;
          sp = p;
          dp = d;
        }
      }
    }
    if (!found) break;
    if (best[0] == bm) plateaus++;
    swaps[2 * nswap] = sp;
    swaps[2 * nswap + 1] = dp;
    nswap++;
    std::swap(m[sp], m[dp]);
    std::swap(q[sp], q[dp]);
    std::swap(origin[sp], origin[dp]);
    recalc();
  }
  if (*std::max_element(bl.begin(), bl.end()) >= baseB) {
    nswap = 0;
    std::fill(swaps, swaps + 32, -1);
    for (int p = 0; p < DP; p++) {
      m[p] = km[p];
      q[p] = kq[p];
      origin[p] = (used && p == ki[3]) ? ki[2] : p;
    }
    recalc();
  }
  int n = 0;
  std::fill(copies, copies + 2 * MCOPIES, -1);
  for (int p = 0; p < DP; p++)
    if (m[p] != oldm[p]) {
      if (n >= MCOPIES) return -11;
      copies[2 * n] = origin[p];
      copies[2 * n + 1] = p;
      n++;
    }
  I t0[DR] = {}, b0[DR] = {};
  for (int p = 0; p < DP; p++) {
    t0[p / DS] += oldq[p];
    b0[p / DS] += (oldq[p] + WAN22_BALANCE_BLOCK_SIZE - 1) / WAN22_BALANCE_BLOCK_SIZE;
    outm[p] = m[p];
    outq[p] = q[p];
  }
  for (int r = 0; r < DR; r++) {
    counts[r] = sent[r];
    counts[DR + r] = received[r];
    if (sent[r] > budget || received[r] > budget) return -12;
  }
  I vals[12] = {
      n,
      nswap,
      *std::max_element(b0, b0 + DR),
      *std::max_element(bl.begin(), bl.end()),
      *std::max_element(t0, t0 + DR),
      *std::max_element(t.begin(), t.end()),
      used,
      budget,
      baseB,
      baseT,
      plateaus,
      loops};
  std::copy(vals, vals + 12, info);
  return n;
}
