# Development replication: frozen v1 models vs v3 (sizes-v1 + buckets-v3) snapshots

Fresh development seed block 800000, 10,000 paired pairs per opponent, six pool-v1 bots, AIVAT, every model on identical
deals (`scripts/dev_compare.sh`). Differences from `python -m cortado.compare` (bb/100, 95% CI clustered by deal, not
corrected for multiple comparisons). Development data only; not a final test.

| Comparison | nit | tag | min-raiser | third-pot | random-size | overbet |
|---|---|---|---|---|---|---|
| v3 100 BB 3.18M − A | +17.0 [−3.5, +37.4] | +20.8 [−5.0, +46.6] | **+37.3 [+10.2, +64.4]** | +27.6 [−0.9, +56.1] | **+36.1 [+5.8, +66.3]** | **+76.2 [+44.3, +108.1]** |
| v3 100 BB 7.78M − A | **+25.8 [+5.9, +45.7]** | +21.3 [−4.6, +47.2] | +13.6 [−14.0, +41.1] | +20.3 [−8.5, +49.1] | +24.2 [−6.1, +54.5] | +17.3 [−14.9, +49.4] |
| v3 100 BB 7.78M − 3.18M | +8.8 [−5.6, +23.3] | +0.5 [−19.3, +20.4] | **−23.7 [−44.2, −3.3]** | −7.3 [−28.7, +14.1] | −11.8 [−34.5, +10.8] | **−58.9 [−80.7, −37.1]** |
| v3 200 BB 4.78M − B | −22.8 [−54.6, +9.1] | +15.0 [−27.7, +57.7] | +24.1 [−23.4, +71.6] | −0.9 [−48.5, +46.7] | +39.0 [−13.3, +91.3] | **+131.5 [+72.9, +190.1]** |
| v3 200 BB 7.22M − B | +1.8 [−29.9, +33.6] | +17.3 [−24.4, +59.1] | +14.9 [−32.5, +62.3] | −18.2 [−64.5, +28.1] | +11.0 [−40.3, +62.4] | **+98.8 [+40.2, +157.4]** |
| v3 200 BB 7.22M − 4.78M | **+24.6 [+6.6, +42.6]** | +2.3 [−25.2, +29.7] | −9.2 [−38.3, +19.9] | −17.3 [−46.9, +12.4] | −28.0 [−59.7, +3.8] | **−32.7 [−63.9, −1.4]** |

Findings:
- No interval shows a frozen v1 model clearly ahead of a v3 snapshot against any bot.
- 100 BB: the 3.18M snapshot beats A clearly against three bots and has positive point estimates against all six.
  Longer training (7.78M) is clearly worse than 3.18M against min-raiser and overbet.
- 200 BB: v3 beats B clearly only against the overbet bot; other differences are undetected.
  7.22M vs 4.78M trades: better against nit, worse against overbet.

Also: real-history re-solving on the 200 BB 4.78M snapshot (`results/rhresolve-ab-200`, seed 700000, 1,500 pairs):
re-solver − blueprint: min-raiser −144.1 [−218.1, −70.0], third-pot −82.1 [−149.6, −14.5], tag −54.0 [−109.5, +1.5],
random-size −38.9, overbet −9.5, nit +1.1 (last three undetected). Confirms the decision to drop re-solving.
