# River re-solving: A/B results (development, not a final test)

Base checkpoint `runs/buckets-42/iteration-525987.json.gz` (v1, 100 BB). Both agents play identical deals
(seed block 720000, 600 paired deals per opponent, AIVAT-corrected). Numbers are the paired change versus
the plain blueprint-plus-translation adapter, bb/100 with 95% CI. Per-opponent, not multiplicity-corrected.
Re-solving is used on 6-15% of the candidate's decisions (river decisions only).

| Opponent | Unsafe re-solve | Safe re-solve (gadget) |
|---|---|---|
| nit | -22 [-72, 28] | -22 [-69, 25] |
| tag | **-71 [-137, -4]** | -26 [-86, 34] |
| min-raiser | -15 [-73, 43] | -15 [-68, 37] |
| third-pot | **-75 [-127, -23]** | -46 [-94, 2] |
| random-size | +4 [-59, 67] | +48 [-13, 108] |
| overbet | +1 [-28, 30] | -2 [-34, 30] |
| p95 river-decision latency | 7-16 s | 8-19 s |

Mirror test (unsafe re-solver vs the blueprint agent itself, 480 pairs): -11 [-63, +41].
Solver checks: fast showdown matches the dense reference to about 1e-13; CFR+ converges below 1% of the pot;
on a polarized toy spot it bluffs 54% (theory 55%).

Conclusion: the safe version removes the significant losses of the unsafe version, but **neither shows a
benefit**. The safe version averages about -10 bb/100 across the six opponents, and its latency is far too high
for live play. Re-solving stays off by default (`--agent blueprint`).

Known approximations that may limit it:
* gadget values come from the abstract blueprint river and are rescaled from abstract to real pot size;
* our own range is estimated through the same lossy bet translation;
* imperfect-recall buckets mean the blueprint's river values are coarse;
* when the pre-river replay desyncs (overbets mapped to all-in), it falls back and never re-solves.
