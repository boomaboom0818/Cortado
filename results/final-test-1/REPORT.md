# Final test 1: reserved deals (pre-registered in EVALUATION_PROTOCOL.md, commit 84075e2)

Base seed 100,000,000 (fresh; no earlier run touched it). 30,000 paired deals per pool bot, 3,000 per LBR run.
AIVAT-corrected bb/100, 95% t-intervals clustered by paired deal. Bonferroni lower bound uses alpha 0.05 / 9.
Each candidate was run once. Candidates were chosen on development data only.

| Opponent | A, 100 BB (526k it.) | B, 200 BB (1.35M it.) |
|---|---|---|
| nit | +5.4 [-4.4, +15.3] | +71.9 [+56.6, +87.1] |
| tag | +113 [+100, +126] | +246 [+225, +268] |
| lag | +262 [+247, +277] | +460 [+434, +486] |
| station | +511 [+494, +529] | +672 [+645, +698] |
| maniac | +627 [+610, +645] | +1,091 [+1,055, +1,126] |
| min-raiser | +159 [+145, +173] | +312 [+289, +336] |
| third-pot | +162 [+148, +176] | +289 [+265, +313] |
| overbet | +226 [+210, +242] | +396 [+366, +426] |
| random-size | +230 [+215, +245] | +408 [+382, +435] |
| LBR | +43 [+12, +74] | +85 [+26, +145] |
| Bonferroni lower bound, worst bot | -8.5 (nit) | +50.3 (nit) |

## Outcome under the pre-registered rules
* **B:** beats all nine pool-v1 bots, and the pooled claim holds (every Bonferroni lower bound > 0, worst case +50 bb/100 vs the nit).
  LBR found no profitable exploit (lower bound +26 > 0; Bonferroni-adjusted bound +1.3).
* **A:** beats eight of nine bots. The nit interval includes zero, so the pooled claim is **not** met. LBR also found no profitable exploit (lower bound +12).
* Trained-table coverage: 93% (overbet) to 100%; rest is the heuristic fallback.
* A and B play different stack depths, so their numbers are not directly comparable.

## Limits
* Pool-v1 bots are scripts we wrote. Beating them is not evidence of strength against strong opponents.
* LBR is a weak approximate best response. Its result does not bound exploitability and supports no GTO claim.
* Raw data: A.json, B.json in this directory.
