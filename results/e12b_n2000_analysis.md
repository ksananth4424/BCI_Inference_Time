# E12b n=2000 Analysis

## Run Summary

- Samples: 2000
- Baseline correct: 1258 (62.9%)
- Corrected correct: 1327 (66.35%)
- Absolute gain: +3.45 percentage points
- Flip-to-correct: 158 (7.9%)

## Paired Significance (Baseline vs E12b Corrected)

- Wins (baseline wrong -> corrected right): 158
- Losses (baseline right -> corrected wrong): 89
- Net wins: +69
- McNemar exact p-value: 1.34e-05

## Intervention Diagnostics

- Intervened rate: 59.85%
- Fallback rate: 0.10%

Interpretation: the tuned confidence gate (0.18) yields a robust, statistically significant improvement at larger scale.

## Positioning vs E1 (n=500 matched split)

Using paired comparison on shared 500-sample split:
- E1 corrected accuracy: 77.2%
- E12b corrected accuracy: 62.4%
- Delta (E12b - E1): -14.8 pp
- McNemar exact p-value: 1.31e-14

Interpretation: E12b is a useful robust selective-intervention variant, but E1 remains the primary strongest method for headline performance.
