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
