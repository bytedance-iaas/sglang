import EicSpec

/-!
Replays a trace recorded by test_eic_lean_spec.py. One step per line, tab-separated:

  label  prims  pre  post  avail evict prot

`prims` is `name:i,j;name:k` (or `-`), applied in order to `pre`. Slots are
`f`, `l` or `c<lock>`; anything else is a slot the implementation lost or
double-owned. A step passes when the spec accepts every transition, lands exactly
on `post`, and the implementation's counters equal the spec's counts of `post`.
-/

open EIC

def parsePage : String → Option Page
  | "f" => some .free
  | "l" => some .loading
  | s => if s.startsWith "c" then (s.drop 1).toString.toNat?.map .cached else none

def parsePool (s : String) : Except String Pool :=
  (s.splitOn " ").filter (· ≠ "") |>.mapM fun t =>
    match parsePage t with
    | some p => .ok p
    | none => .error s!"unrepresentable slot state '{t}'"

def transition : String → Option (Page → Option Page)
  | "lock" => some lockF
  | "unlock" => some unlockF
  | "startLoad" => some startLoadF
  | "ackOk" => some ackOkF
  | "ackFail" => some ackFailF
  | "insert" => some insertF
  | "evict" => some evictF
  | _ => none

def applyPrims (p : Pool) (prims : String) : Except String Pool := do
  if prims == "-" then return p
  let mut cur := p
  for prim in prims.splitOn ";" do
    let [name, idx] := prim.splitOn ":" | throw s!"bad prim '{prim}'"
    let some f := transition name | throw s!"unknown transition '{name}'"
    let is ← (idx.splitOn ",").mapM fun t =>
      match t.toNat? with
      | some n => .ok n
      | none => .error s!"bad slot index '{t}'"
    match stepAll f cur is with
    | some q => cur := q
    | none => throw s!"spec rejects {name} on slots {is}"
  return cur

def checkStep (line : String) : Except String Unit := do
  let [_, prims, pre, post, counts] := line.splitOn "\t"
    | throw "expected 5 tab-separated fields"
  let pre ← parsePool pre
  let post ← parsePool post
  let got ← applyPrims pre prims
  unless got == post do
    throw s!"implementation reached {repr post}, spec reached {repr got}"
  let [a, e, q] := (counts.splitOn " ").filterMap String.toNat?
    | throw s!"bad counters '{counts}'"
  unless (a, e, q) == (avail post, evict post, prot post) do
    throw s!"counters avail/evict/prot = {a}/{e}/{q}, spec = {avail post}/{evict post}/{prot post}"

def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: eic_replay <trace>"; return 2
  let lines := (← IO.FS.lines path).toList.filter (· ≠ "")
  for (line, n) in lines.zip (List.range lines.length) do
    match checkStep line with
    | .ok () => pure ()
    | .error msg =>
      IO.eprintln s!"step {n + 1} [{(line.splitOn "\t").head!}]: {msg}"
      return 1
  IO.println s!"ok: {lines.length} steps match the spec"
  return 0
