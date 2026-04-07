# E12 vs E12b (Paired, n=500)

## Headline

- E12b (min_intervention_score=0.18) improves over E12 (0.25) on the same 500 sampled questions.

## Metrics

- E12 corrected accuracy: 59.6% (298/500)
- E12b corrected accuracy: 62.4% (312/500)
- Absolute gain: +2.8 percentage points
- E12 flip-to-correct: 5.0% (25/500)
- E12b flip-to-correct: 9.8% (49/500)

## Paired Outcome Analysis

- Wins (E12 wrong -> E12b right): 25
- Losses (E12 right -> E12b wrong): 11
- Net wins: +14
- McNemar exact p-value: 0.0288

## Intervention Behavior

- E12 intervened on 35.2% of samples
- E12b intervened on 58.8% of samples

Interpretation: lowering the intervention threshold from 0.25 to 0.18 substantially increases effective corrections and yields a statistically significant net improvement over the stricter gate.

## Recommendation

- Keep E12b as the preferred confidence-gated setting.
- Next run: replicate E12b on n=2000 for stable final reporting and compare against strongest ungated E1 variant in final tables.
