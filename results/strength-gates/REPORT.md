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
