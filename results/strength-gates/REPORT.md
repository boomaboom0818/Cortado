# Strength gates: development results (not a final test)

Candidates: `buckets-v1` FCPA MCCFR, 100 BB, seeds 42/43/44, about 160k iterations each (60-minute CPU budget,
~330 MB RSS, ~43.5k information sets, table growth saturated). Played in the **unrestricted-size game** through
`FullActionAgent` (pseudo-harmonic translation). Seed block 700000 (development). 3,000 paired deals (6,000
hands) per pool opponent, 300 pairs vs LBR (60 sampled victim hands per decision). Per-opponent 95% t-intervals
over paired clusters, not multiplicity-corrected. Latency is the candidate's own decision time.

| Opponent (pool-v1) | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| nit | -70 [-122, -17] | -104 [-155, -53] | -51 [-101, -1] |
| tag | 38 [-32, 107] | 28 [-41, 96] | 36 [-32, 104] |
| lag | 224 [137, 311] | 223 [138, 308] | 207 [122, 293] |
| station | 720 [617, 823] | 698 [598, 797] | 698 [597, 799] |
| maniac | 644 [526, 763] | 621 [504, 738] | 667 [550, 784] |
| min-raiser | 33 [-40, 107] | 43 [-31, 117] | 10 [-63, 84] |
| third-pot | 88 [14, 163] | 106 [32, 179] | 115 [41, 190] |
| overbet | 200 [119, 282] | 236 [156, 316] | 190 [108, 272] |
| random-size | 143 [65, 221] | 113 [36, 190] | 105 [25, 184] |
| **LBR** | **-211 [-416, -5]** | **-199 [-387, -12]** | -35 [-235, 165] |

Coverage: 99-100% of candidate decisions come from the trained table against most opponents (previous exact/suit
tables: ~53% uniform fallback vs a calling station, 100% on turn and river). Against `overbet` ~7% and
`random-size` ~3% of decisions fall back to an equity heuristic after the abstract replay desynchronises
(mapped all-ins). Candidate decision latency p95 ≈ 2.6-3.4 ms. Each seed's full benchmark: ~55 s wall on 10 processes.

What this shows: the coverage problem is fixed, full-action play works without illegal actions, and results are
consistent across seeds. What it does not show: competitive strength. The candidate **loses to the nit script on
all three seeds** and is **exploited by LBR** on two of three seeds. Pool opponents are diagnostic scripts written in
this repository, not independent strong agents. No external opponent has been played.


## Longer training and board texture (AIVAT-corrected, development seed block 700000)

bb/100 ± 95% half-width, AIVAT weights (1,1,1,1); 3,000 paired deals per pool opponent, 300 vs LBR. Same deals for every column.
`v1` = buckets-v1, `v2` = buckets-v2 (board texture); 100/200 = stack in BB; k = MCCFR iterations.

| Opponent | v1-100-526k | v2-100-s42-441k | v2-100-s43-430k | v1-200-155k | v1-200-527k | v2-200-416k |
|---|---|---|---|---|---|---|
| nit | 5 ± 30 | -75 ± 37 | -90 ± 38 | -134 ± 71 | 23 ± 57 | -160 ± 72 |
| tag | 104 ± 42 | -17 ± 47 | -9 ± 48 | 9 ± 84 | 184 ± 77 | -38 ± 91 |
| lag | 291 ± 48 | 213 ± 56 | 212 ± 54 | 286 ± 103 | 483 ± 90 | 296 ± 104 |
| station | 563 ± 54 | 716 ± 72 | 716 ± 69 | 1424 ± 136 | 1108 ± 112 | 1353 ± 138 |
| maniac | 592 ± 53 | 615 ± 54 | 620 ± 55 | 1300 ± 110 | 1276 ± 107 | 1364 ± 114 |
| min-raiser | 108 ± 42 | 29 ± 49 | 69 ± 50 | 76 ± 88 | 245 ± 78 | 75 ± 96 |
| third-pot | 170 ± 46 | 87 ± 50 | 89 ± 50 | 203 ± 94 | 377 ± 82 | 257 ± 94 |
| overbet | 235 ± 50 | 276 ± 52 | 295 ± 54 | 400 ± 100 | 448 ± 98 | 521 ± 99 |
| random-size | 220 ± 46 | 189 ± 49 | 167 ± 50 | 249 ± 96 | 386 ± 83 | 318 ± 95 |
| lbr | 54 ± 93 | 40 ± 90 | 26 ± 113 | -27 ± 186 | 48 ± 128 | 114 ± 176 |
| LBR raw (no AIVAT) | -17 ± 178 | 42 ± 185 | 10 ± 187 | 162 ± 377 | -123 ± 237 | 276 ± 348 |
| min table fraction | 0.931 | 0.936 | 0.932 | 0.918 | 0.922 | 0.926 |

Observations (development data, per-opponent intervals, not multiplicity-corrected):

* Longer training helps: v1 100 BB went from losing to `nit` at 164k (raw -70) to about break-even at 526k (+5 ± 30), and improved against tag, min-raiser, third-pot and random-size. The 200 BB checkpoint improved similarly from 155k to 527k.
* Board texture (v2) did **not** help at this compute: about 5x more information sets (213k vs 43.6k), so each is visited less and the average strategy is less converged. v2 is worse than v1 at a similar iteration count against nit/tag. This is a statement about this training budget, not about texture features in general.
* The LBR exploiter no longer shows a significant edge against the longest-trained checkpoints, but its intervals are wide (300 pairs) and LBR is a myopic lower bound: failing to exploit does not show low exploitability.
* AIVAT variance ratio averaged 0.39-0.47 across opponents.
* Table coverage is lower than the per-opponent maximum suggests against `overbet` (~92-94%): the rest is desync fallback after mapped all-ins.
