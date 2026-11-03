#pragma once

// Required specialization parameters come from MoeShape.cuda_flags.
#if !defined(WAN22_LOGICAL_EXPERTS) || !defined(WAN22_PHYSICAL_EXPERTS) || !defined(WAN22_EP_SIZE) || \
    !defined(WAN22_HIDDEN_SIZE) || !defined(WAN22_TOP_K)
#error "Supply model geometry through MoeShape"
#endif
namespace {
constexpr int DL = WAN22_LOGICAL_EXPERTS, DP = WAN22_PHYSICAL_EXPERTS;
constexpr int DR = WAN22_EP_SIZE, DS = DP / DR, DC = DP - DL + 1;
constexpr int DREP = (DP == DL) ? 1 : ((2 * (DP - DL) < DP) ? 2 * (DP - DL) : DP);
constexpr int DPAIRS = DC > 1 ? DC*(DC - 1) / 2 : 1;
constexpr int DN = DL * DREP;
constexpr int NS = 0, SL = DL, LT = SL + DL * DC, TT = LT + DL, BB = TT + DR, RP = BB + DR, NR = RP + DREP,
              DWORK = NR + 1;
constexpr int MCOPIES = 4 * DR, MSNAP = 12 + 2 * MCOPIES + 32;
constexpr int WM = 0, WQ = DP, WO = 2 * DP, WBM = 3 * DP, WBQ = 4 * DP, WOLD = 5 * DP, WT = 6 * DP;
constexpr int WB = WT + DR, WS = WB + DR, WR = WS + DR, WMASK = WR + DR, WHOT = WMASK + DL;
constexpr int WNH = WHOT + DP, WMAX = WNH + 1, WNMAX = WNH + 2, WB2 = WNH + 3, WBASEB = WNH + 4, WBASET = WNH + 5;
constexpr int WNS = WNH + 6, WPL = WNH + 7, WL = WNH + 8, WA = WNH + 9, MWORK = WA + 1;
static_assert(DL > 0 && DP >= DL && DR > 0 && DR <= 32 && DP % DR == 0);
}  // namespace
