# Phase 1 — Compiler foundations

This worked example fixes the conventions that the Phase 2 generator must
implement.  Each assignment defines its left-hand-side variable and each
operand (including `return`'s operand) is a use.  The control-flow graph is a
straight line, so instruction *i* has instruction *i + 1* as its only
successor; `return` has no successor.

## Three-address code

```text
1: a = 1
2: b = 2
3: c = a + b
4: d = a * c
5: e = b + d
6: return e
```

## Backward liveness

The transfer equations are:

```text
live_out[i] = union(live_in[s] for s in successors(i))
live_in[i]  = use[i] union (live_out[i] - def[i])
```

Applying them from line 6 back to line 1 gives:

| Line | `def` | `use` | `live_out` | `live_in` |
|---:|:---:|:---:|:---:|:---:|
| 1 `a = 1` | `{a}` | `{}` | `{a}` | `{}` |
| 2 `b = 2` | `{b}` | `{}` | `{a,b}` | `{a}` |
| 3 `c = a + b` | `{c}` | `{a,b}` | `{a,b,c}` | `{a,b}` |
| 4 `d = a * c` | `{d}` | `{a,c}` | `{b,d}` | `{a,b,c}` |
| 5 `e = b + d` | `{e}` | `{b,d}` | `{e}` | `{b,d}` |
| 6 `return e` | `{}` | `{e}` | `{}` | `{e}` |

For example, line 4 has `live_out = {b,d}` because that is line 5's
`live_in`.  Its definition kills `d`, and its uses add `a` and `c`, yielding
`live_in = {a,b,c}`.

## Interference graph

For every definition `x` at a program point, add an undirected interference
edge `(x,y)` for every `y` in that instruction's `live_out`, excluding `x`
itself.  The resulting edges are:

| Defining line | Edges added |
|---:|:---|
| 1 | none |
| 2 | `b--a` |
| 3 | `c--a`, `c--b` |
| 4 | `d--b` |
| 5 | none (`e` is only live with itself) |
| 6 | none |

```text
    a---b---d       e
     \\\ /
      c

Edges: {a-b, a-c, b-c, b-d}; e is isolated.
```

Thus `{a,b,c}` is a three-node clique, `d` has degree 1, and `e` has degree 0.
This example has no copy/move instruction, so the move-edge set is
empty and there are no coalescing opportunities.

## Chaitin–Briggs walk-through (`k = 2` registers)

Use registers `R1` and `R2`.  Simplify first removes every node whose
current degree is less than `k`:

1. Remove `e` (degree 0) and push it.
2. Remove `d` (degree 1) and push it.  The remaining graph is the triangle
   `{a,b,c}`, where every node has degree 2, so simplify is stuck.
   Compute `spill_cost = (uses + defs) / degree`:

   | Node | Uses + defs | Degree | Spill cost |
   |:---:|---:|---:|---:|
   | `a` | 3 | 2 | 1.50 |
   | `b` | 3 | 2 | 1.50 |
   | `c` | 2 | 2 | 1.00 |

   Choose `c` as the cheap potential spill and push it.
3. Removing `c` leaves edge `{a,b}`.  Both nodes now have degree 1, so remove
   and push `a`, then `b`.

The bottom-to-top stack is `e, d, c*, a, b`, where `c*` is a potential spill.
On popping, one valid colouring is:

| Pop | Forbidden colours from already-coloured neighbours | Result |
|:---:|:---|:---|
| `b` | none | `b = R1` |
| `a` | `R1` (`b`) | `a = R2` |
| `c*` | `R1`, `R2` (`b`, `a`) | spill `c` |
| `d` | `R1` (`b`) | `d = R2` |
| `e` | none | `e = R1` |

`c` really must spill for this 2-register graph: it belongs to a clique of
size three.  A spill rewrite would insert stores after `c`'s definition and
reloads before its uses, then rerun liveness and allocation.

The Briggs conservative coalescing rule is also part of the generator's
specification: for a move pair, merge only when the merged node has fewer than
`k` neighbours of significant degree (degree at least `k`).  There are no move
edges here, so no merge is attempted.  Future generated examples must record
move edges separately from interference edges and apply this check before
coalescing.
