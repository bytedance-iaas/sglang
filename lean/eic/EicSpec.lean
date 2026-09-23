/-
Executable spec of EIC device-slot ownership (EICPagedHiRadixCache).

Every device slot is free, loading (load-back DMA not yet acked) or cached with a
lock count. Each cache operation is a batch of per-slot transitions; a transition
returning `none` is one the implementation must never perform. `Main.lean` replays
traces recorded from the Python implementation against these transitions
(test_eic_lean_spec.py). Theorem names cite the PR whose bug they rule out.
-/
namespace EIC

/-- Owner of one device slot; `cached lock` counts the holders. -/
inductive Page where
  | free
  | loading
  | cached (lock : Nat)
deriving DecidableEq, Repr

abbrev Pool := List Page

def isFree : Page → Bool | .free => true | _ => false
def isEvict : Page → Bool | .cached 0 => true | _ => false
def isProt : Page → Bool | .loading => true | .cached (_+1) => true | _ => false

def avail (p : Pool) : Nat := p.countP isFree
def evict (p : Pool) : Nat := p.countP isEvict
def prot  (p : Pool) : Nat := p.countP isProt

/-- The idle leak check: every slot is in exactly one of the three counters. -/
theorem conservation (p : Pool) : avail p + evict p + prot p = p.length := by
  induction p with
  | nil => rfl
  | cons x xs ih =>
    simp only [avail, evict, prot, List.countP_cons, List.length_cons] at *
    cases x with
    | free => simp [isFree, isEvict, isProt]; omega
    | loading => simp [isFree, isEvict, isProt]; omega
    | cached k => cases k <;> simp [isFree, isEvict, isProt] <;> omega

/-- Apply a transition to slot `i`; `none` means the operation is illegal. -/
def step (f : Page → Option Page) (p : Pool) (i : Nat) : Option Pool :=
  match p[i]? with
  | some x => (f x).map (p.set i ·)
  | none => none

def lockF : Page → Option Page | .cached k => some (.cached (k+1)) | _ => none
def unlockF : Page → Option Page | .cached (k+1) => some (.cached k) | _ => none
def startLoadF : Page → Option Page | .free => some .loading | _ => none
def ackOkF : Page → Option Page | .loading => some (.cached 0) | _ => none
def ackFailF : Page → Option Page | .loading => some .free | _ => none
def insertF : Page → Option Page | .free => some (.cached 0) | _ => none
def evictF : Page → Option Page | .cached 0 => some .free | _ => none

/-- A batch is illegal if any of its slot transitions is. -/
def stepAll (f : Page → Option Page) (p : Pool) : List Nat → Option Pool
  | [] => some p
  | i :: is => (step f p i).bind (stepAll f · is)

/-- #748: a node with lock_ref 0 must not be dec_lock_ref'd. -/
theorem unlock_unheld_rejected (p : Pool) (i : Nat) (h : p[i]? = some (.cached 0)) :
    step unlockF p i = none := by simp [step, h, unlockF]

/-- #768: a slot still loading cannot be adopted by a same-prefix request. -/
theorem adopt_loading_rejected (p : Pool) (i : Nat) (h : p[i]? = some .loading) :
    step lockF p i = none := by simp [step, h, lockF]

/-- #768: a failed load frees only loading slots, never a cached one. -/
theorem ackFail_held_rejected (p : Pool) (i k : Nat) (h : p[i]? = some (.cached k)) :
    step ackFailF p i = none := by simp [step, h, ackFailF]

theorem countP_set {α} (q : α → Bool) (l : List α) (i : Nat) (a b : α)
    (h : l[i]? = some a) :
    l.countP q + (if q b then 1 else 0) = (l.set i b).countP q + (if q a then 1 else 0) := by
  induction l generalizing i with
  | nil => simp at h
  | cons x xs ih =>
    cases i with
    | zero => simp at h; subst h; simp [List.countP_cons]; omega
    | succ j => simp at h; have := ih j h; simp [List.countP_cons]; omega

/-- Eviction never takes a locked slot. -/
theorem evict_held_rejected (p : Pool) (i k : Nat) (h : p[i]? = some (.cached (k+1))) :
    step evictF p i = none := by simp [step, h, evictF]

/-- Eviction never takes a slot whose load is in flight. -/
theorem evict_loading_rejected (p : Pool) (i : Nat) (h : p[i]? = some .loading) :
    step evictF p i = none := by simp [step, h, evictF]

/-- A load takes one free slot and leaves evictable unchanged. -/
theorem startLoad_counts (p p' : Pool) (i : Nat) (h : step startLoadF p i = some p') :
    avail p' + 1 = avail p ∧ evict p' = evict p := by
  unfold step at h
  split at h
  · rename_i x hx
    cases x <;> simp [startLoadF] at h
    subst h
    have ha := countP_set isFree p i .free .loading hx
    have he := countP_set isEvict p i .free .loading hx
    simp [isFree, isEvict] at ha he
    exact ⟨by simp [avail]; omega, by simp [evict]; omega⟩
  · simp at h

def startLoads (p : Pool) (is : List Nat) : Option Pool := stepAll startLoadF p is

theorem startLoads_counts (p p' : Pool) (is : List Nat) (h : startLoads p is = some p') :
    avail p' + is.length = avail p ∧ evict p' = evict p := by
  induction is generalizing p with
  | nil => simp [startLoads, stepAll] at h; subst h; simp
  | cons i is ih =>
    simp only [startLoads, stepAll, Option.bind_eq_some_iff] at h
    obtain ⟨m, hm, hr⟩ := h
    have h1 := startLoad_counts p m i hm
    have h2 := ih m hr
    simp; omega

/-- #741: the admission gate keeps `reserve` slots obtainable after the load. -/
def admit (p : Pool) (is : List Nat) (reserve : Nat) : Option Pool :=
  if avail p + evict p ≥ is.length + reserve then startLoads p is else none

theorem admit_keeps_reserve (p p' : Pool) (is : List Nat) (r : Nat)
    (h : admit p is r = some p') : avail p' + evict p' ≥ r := by
  unfold admit at h
  split at h
  · have := startLoads_counts p p' is h; omega
  · simp at h

/-- #799: read credits; free + held is conserved, over-release is refused. -/
structure Credit where
  free : Nat
  held : Nat

def acquire (c : Credit) (n : Nat) : Option Credit :=
  if n ≤ c.free then some ⟨c.free - n, c.held + n⟩ else none

def release (c : Credit) (n : Nat) : Option Credit :=
  if n ≤ c.held then some ⟨c.free + n, c.held - n⟩ else none

theorem acquire_conserves (c c' : Credit) (n : Nat) (h : acquire c n = some c') :
    c'.free + c'.held = c.free + c.held := by
  unfold acquire at h; split at h <;> simp at h; subst h; simp; omega

theorem release_conserves (c c' : Credit) (n : Nat) (h : release c n = some c') :
    c'.free + c'.held = c.free + c.held := by
  unfold release at h; split at h <;> simp at h; subst h; simp; omega

end EIC
