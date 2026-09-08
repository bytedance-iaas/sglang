"""Forward-boundary rendezvous for SiDP Direction A dynamic nptr (Phase 2b).

The coordinated device barrier needs every participating member to agree, each
forward, on how many members will arrive (nptr). Members are independent
processes with no shared clock, so this is a host-side consensus taken at the
one point every member already passes each iteration: right after the scheduler
decides this iteration's batch. Each member votes decode(1)/idle(0); the live
participant count is the sum. Only decode voters run the forward + barrier (which
waits for exactly that many arrivals), so a member that just hit EOS removes
itself by voting 0 -- an explicit exit, never a guess.

Deadlock-free by unconditional all-vote
---------------------------------------
Once coordinated mode is on, EVERY member votes EVERY scheduler iteration and
waits for ALL members' votes for that round. Because everyone always votes, the
per-member round counter never drifts: ``wait(all)`` blocks a fast member until
the slowest has voted for the same round. (This is the proven ForceSyncStrategy
mechanism; its only flaw was idle ranks dropping out -- here idle ranks vote 0.)
A scheduler loop keeps returning to this point even after its requests hit EOS
(batch=None just routes to the idle branch and loops again), so an exited member
keeps voting 0 until new work arrives.

Cost: members keep voting even when the whole cohort is idle. To avoid a busy
CPU spin, the caller sleeps briefly after an all-zero round (handled by the
scheduler's idle path); the rendezvous itself only adds one set + one wait per
iteration, which is negligible next to a forward.

Consensus rides the existing SiDP TCPStore (member 0 is master). Keys:
    sidp/coord/vote/{round}/{rank} -> member r's vote (0/1) for this round
    sidp/coord/nptr/{round}        -> member 0's authoritative live nptr
"""

from __future__ import annotations

from datetime import timedelta

# Unified-schedule group forward modes (see rendezvous_unified).
UNIFIED_DECODE = "d"
UNIFIED_PREFILL = "p"
UNIFIED_IDLE = "i"


class CoordRendezvous:
    """Per-forward decode/idle consensus that yields the live barrier nptr."""

    def __init__(
        self,
        *,
        store,
        member_rank: int,
        world_size: int,
        timeout_s: float = 30.0,
    ) -> None:
        self.store = store
        self.rank = member_rank
        self.world = world_size
        self.timeout = timedelta(seconds=timeout_s)
        # Monotonic round counter, kept aligned across members by wait(all).
        self._round = 0
        self.last_round = -1  # round used by the most recent rendezvous() call

    def _get_int(self, key: str) -> int:
        raw = self.store.get(key)
        return int(raw.decode() if isinstance(raw, bytes) else raw)

    def rendezvous(self, wants_forward: bool) -> tuple[bool, int]:
        """Reach consensus for this forward boundary.

        Returns ``(participate, live_nptr)``:
          * ``participate`` -- whether THIS member should run its forward now.
          * ``live_nptr``   -- members that will arrive at the barrier this round
            (>=1 when participate is True). 0 means the whole cohort is idle.

        Must be called every scheduler iteration by every member while
        coordinated mode is on, so round numbers stay aligned.
        """
        rnd = self._round
        self.last_round = rnd  # expose the round used (read-only, for observers)
        vote = 1 if wants_forward else 0
        self.store.set(f"sidp/coord/vote/{rnd}/{self.rank}", str(vote))
        keys = [f"sidp/coord/vote/{rnd}/{r}" for r in range(self.world)]
        try:
            self.store.wait(keys, self.timeout)
        except RuntimeError as exc:
            raise RuntimeError(
                f"SiDP rendezvous: member {self.rank} timed out at round {rnd} "
                f"waiting for all {self.world} votes"
            ) from exc

        nptr_key = f"sidp/coord/nptr/{rnd}"
        if self.rank == 0:
            total = sum(
                self._get_int(f"sidp/coord/vote/{rnd}/{r}")
                for r in range(self.world)
            )
            self.store.set(nptr_key, str(total))
        else:
            self.store.wait([nptr_key], self.timeout)
        live = self._get_int(nptr_key)

        self._round = rnd + 1
        # A member only participates in the barrier if it is forwarding; live is
        # the number of forwarding members this round (what the barrier waits on).
        return (wants_forward and live > 0, live)

    def rendezvous_unified(
        self, *, want_prefill: bool, has_running: bool, chunked_must: bool,
        below_watermark: bool,
    ) -> str:
        """Unified-schedule consensus: agree on ONE group forward mode this round.

        Runs ALONGSIDE the plain ``rendezvous`` (not instead of it): this decides
        the group mode M (intent), while the barrier nptr is set by the plain
        rendezvous later, once the real decode set is known. So this does NOT
        return or compute a live count -- that would risk a count/arrival mismatch
        (a member intending decode can still finalize an empty decode batch).

        Every member reports its cheap intent (already computed inside the prefill
        decision), member 0 arbitrates a single group mode M, and everyone reads
        it -- so prefill/decode never mix across ranks.

        Intent payload per member: ``"<intent>,<has_running>,<below_wm>,<chunked>"``
        where intent is d(ecode-capable, wants decode)/p(refill wanted)/i(dle).

        Arbitration (decode-wins with anti-starvation), member 0:
          * any rank wants prefill but has no running  -> PREFILL (else it idles)
          * any rank wants prefill and below watermark -> PREFILL (soft refill)
          * any rank can decode                        -> DECODE  (decode-wins)
          * any rank wants prefill                     -> PREFILL (natural align)
          * else                                       -> IDLE

        ``chunked_must`` does NOT change M (a rank mid chunked-prefill must finish
        its prefill regardless; it just reports so, and runs prefill locally even
        when M==DECODE -- the one unavoidable mixed round, see design §2.3).

        Returns M in {"d","p","i"}.
        """
        rnd = self._round
        self.last_round = rnd
        if chunked_must:
            intent = UNIFIED_PREFILL
        elif want_prefill:
            intent = UNIFIED_PREFILL
        elif has_running:
            intent = UNIFIED_DECODE
        else:
            intent = UNIFIED_IDLE
        payload = f"{intent},{int(has_running)},{int(below_watermark)},{int(chunked_must)}"
        self.store.set(f"sidp/coord/uvote/{rnd}/{self.rank}", payload)
        keys = [f"sidp/coord/uvote/{rnd}/{r}" for r in range(self.world)]
        try:
            self.store.wait(keys, self.timeout)
        except RuntimeError as exc:
            raise RuntimeError(
                f"SiDP unified rendezvous: member {self.rank} timed out at round "
                f"{rnd} waiting for all {self.world} intents"
            ) from exc

        mode_key = f"sidp/coord/umode/{rnd}"
        if self.rank == 0:
            intents, runnings, below = [], [], []
            for r in range(self.world):
                raw = self.store.get(f"sidp/coord/uvote/{rnd}/{r}")
                s = raw.decode() if isinstance(raw, bytes) else raw
                it, hr, bw, _ck = s.split(",")
                intents.append(it)
                runnings.append(int(hr))
                below.append(int(bw))
            want_pf = [it == UNIFIED_PREFILL for it in intents]
            can_dec = [runnings[r] == 1 for r in range(self.world)]
            if any(want_pf[r] and runnings[r] == 0 for r in range(self.world)):
                m = UNIFIED_PREFILL  # someone has no decode work -> must prefill
            elif any(want_pf[r] and below[r] == 1 for r in range(self.world)):
                m = UNIFIED_PREFILL  # soft low-watermark refill
            elif any(can_dec):
                m = UNIFIED_DECODE   # decode-wins
            elif any(want_pf):
                m = UNIFIED_PREFILL  # nobody can decode -> natural prefill
            else:
                m = UNIFIED_IDLE
            self.store.set(mode_key, m)
        else:
            self.store.wait([mode_key], self.timeout)
        raw = self.store.get(mode_key)
        m = raw.decode() if isinstance(raw, bytes) else raw

        self._round = rnd + 1
        return m
