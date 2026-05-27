# Sheet 4 — Model Testing and Validation

## Exercise 4.6 — Test Suite from Safety Constraints

### Reference: Safety Constraints from Exercise 2.7

| UCA | Safety Constraint | Level | Verification |
|---|---|---|---|
| UCA-1 | The planner must not issue "continue driving" when a pedestrian is detected within critical stopping distance | System-level | Test planner logic against all detector output combinations |
| UCA-2 | Pedestrian detector must achieve recall >= 99% within ODD — missing a pedestrian is the highest severity failure | Model-level | Test recall on held-out test set |
| UCA-3 | Detection latency must not exceed maximum allowable threshold — 99th percentile latency < 100ms | Model-level | Measure end-to-end inference latency |
| UCA-4 | Emergency brake must remain active until pedestrian/vehicle clears critical distance | System-level | Test planner brake release logic against sustained detector outputs |
| UCA-5 | Traffic light detector must achieve sufficient recall at mapped intersections — missed detection must never be treated as light absent | Model-level | Test traffic light detector recall on intersection scenarios |
| UCA-6a | Perception models must flag ODD violations — confidence must drop measurably on out-of-distribution inputs | Model-level | ODD detection benchmark on rain/fog/night samples |
| UCA-6b | Perception model uncertainty must be well-calibrated within ODD — overconfident wrong predictions must be minimised | Model-level | ECE calibration evaluation |
| UCA-6c | System must detect ODD violations and trigger safe fallback before issuing further driving commands | System-level | Audit fallback behaviour under simulated ODD violations |
| UCA-7 | Operator must be alerted immediately when perception outputs are uncertain — auditory and visual alerts both required | System-level | Audit dashboard alert design |
| UCA-8 | Alert must be issued early enough for human intervention to be effective at current speed | System-level | Calculate minimum warning time at 50 km/h |

The test cases below cover only the model-level constraints (UCA-2, UCA-3, UCA-5,
UCA-6a, UCA-6b). System-level constraints require testing the planner or operator
system and are out of scope for this exercise.

| Constraint ID | Constraint                                                                                          | Test input description                                                                                                 | Expected output                         | Pass criterion                                                                       |
| ------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------ |
| SC-2          | Pedestrian detector must achieve recall >= 99% within ODD                                           | Run the pedestrian detector on the full sunny test split (3,600 images) and compare predictions to ground truth labels | Recall score on the test split          | Recall >= 0.99. The current model achieves 0.599 so this constraint is not satisfied |
| SC-3          | Detection latency must not exceed the maximum allowable threshold — 99th percentile latency < 100ms | Run all three models on 1,000 test images and measure inference time per image                                         | Inference time per image for each model | 99th percentile latency < 100ms for all three models                                 |
| SC-5          | Traffic light detector must achieve sufficient recall at mapped intersections                       | Run the traffic light detector on intersection test frames and compare predictions to ground truth                     | Recall score on intersection frames     | Recall >= 0.99                                                                       |
| SC-6a         | Model confidence must drop measurably on out-of-distribution inputs                                 | Run all three models on the fog and night test splits and compare confidence scores to the sunny baseline              | Mean confidence per split per model     | Mean confidence on fog and night is measurably lower than on the sunny test split    |
| SC-6b         | Perception model uncertainty must be well-calibrated within ODD                                     | Run all three models on the sunny test split and compute Expected Calibration Error per model                          | ECE score per model                     | ECE <= 0.05 for all three models                                                     |

The pedestrian detector (recall = 0.599) and traffic light detector (recall = 0.974)
both fail their respective recall constraints.

---

## Exercise 4.7 — Per-Class Evaluation

**4.7.1 — Precision, Recall, and F1-score per model (test split)**

Computed using sheet03/evaluate.py on the sunny daytime test split (3,600 images).

| Model | Precision | Recall | F1-score |
|---|---|---|---|
| Pedestrian | 0.290 | 0.599 | 0.391 |
| Traffic Light | 0.947 | 0.974 | 0.961 |
| Vehicle | 0.986 | 0.799 | 0.883 |

**4.7.2 — Confusion matrices**

See sheet03/confusion_matrices.png for the full confusion matrices.

| Model | TP | FP | FN | TN |
|---|---|---|---|---|
| Pedestrian | 423 | 1035 | 283 | 1859 |
| Traffic Light | 2518 | 140 | 66 | 876 |
| Vehicle | 2158 | 31 | 542 | 869 |

**4.7.3 — Which model has the lowest recall?**

The pedestrian detector has the lowest recall at 0.599. This means it misses 40% of
real pedestrians in the test set (283 out of 706 positive frames).

This is exactly what the hazard analysis predicted. In Exercise 2.7, UCA-2 identified
missed pedestrian detection as the highest severity failure mode. The reasons are
consistent with the dataset exploration from Sheet 3: pedestrians appear in only 24%
of training frames, occupy a median of just 156 pixels per image, and the model
overfitted after a single epoch due to insufficient positive examples. The traffic
light model, by contrast, benefits from 73% positive class representation and a
consistent visual appearance, which is why it achieves recall of 0.974.

**4.7.4 — Minimum recall required for pedestrian detector before deployment**

A minimum recall of 0.99 is required before the pedestrian detector can be considered
for deployment.

The justification is based on the stopping distance argument. At 50 km/h the vehicle
travels approximately 14 metres per second. A standard emergency braking distance at
this speed is roughly 25 metres, meaning the detector must trigger braking at least
1.8 seconds before impact. If the detector misses 1 in 100 pedestrians (recall = 0.99),
each missed detection is a potential collision with no fallback from the planner since
the planner only acts on detector outputs.

The human operator is the only remaining fallback, but as noted in the system
description, operators work 4-hour shifts and are subject to attention fatigue. Relying
on operator intervention to compensate for a 40% miss rate (the current situation) is
not an acceptable safety argument. The constraint of recall >= 0.99 was derived in
Exercise 2.7 and the current model fails it by a significant margin. The model needs
substantially more training data, particularly positive pedestrian examples, before it
can be considered safe for deployment.
