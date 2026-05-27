# Sheet 3 — Binary Image Classifiers on CARLA Data

## Exercise 3.4 — Dataset Exploration

**3.4.1 — How many images are in the training and test splits?**

| Split      | Images |
|------------|--------|
| Train      | 7,200  |
| Validation | 3,600  |
| Test       | 3,600  |

**3.4.2 — Class distribution. Are classes balanced?**

Training split:

| Label         | Positive | % Positive | Balanced? |
|---------------|----------|------------|-----------|
| Pedestrian    | 1,718    | 23.9%      | No — minority class |
| Traffic Light | 5,276    | 73.3%      | No — majority positive |
| Vehicle       | 5,458    | 75.8%      | No — majority positive |

The pedestrian label is the most imbalanced: only 1 in 4 frames contains a pedestrian.
A naive model that always predicts "absent" would achieve 76% accuracy while being
completely unsafe — it would never detect a pedestrian. This makes **recall** the
critical metric for the pedestrian detector, not accuracy.

**3.4.3 — Example images and patterns**

See `sheet03/sample_images.png` for visual examples of all 8 label combinations.

Key observations:
- Pedestrians are small and distant (median 156 px out of 480,000 total) — making
  them the hardest class to detect.
- Vehicles dominate the frame when present (median 781 px), which is why the vehicle
  classifier is expected to perform best.
- Traffic lights appear at consistent positions (upper half of frame, near intersections)
  and have a characteristic shape — the model can likely learn this positional prior.
- Most frames are intersection scenes with all three classes present simultaneously.
  The rarest combination is pedestrian-only (no vehicle, no traffic light).

---

## Exercise 3.5 — Training the Classifiers

**3.5.1 — Model architecture and training setup**

**Architecture:** ResNet-18 pretrained on ImageNet, with the final fully-connected layer
replaced: `Linear(512 → 1)`. The backbone (17 layers) retains ImageNet pretrained weights.
The single output is a raw logit — no sigmoid applied inside the model, as
`BCEWithLogitsLoss` fuses sigmoid + loss for numerical stability.

**Training setup:**

| Hyperparameter | Value |
|---|---|
| Loss function | BCEWithLogitsLoss (with per-model `pos_weight`) |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| Epochs | 10 |
| Batch size | 32 |
| Input size | 224 × 224 (ImageNet standard) |
| Augmentation | Random horizontal flip (training only) |
| Normalisation | ImageNet mean/std |

Class imbalance was handled via `pos_weight = n_negative / n_positive`:
- Pedestrian: pos_weight = 3.19 (strong upweighting of positive class)
- Traffic light: pos_weight = 0.36
- Vehicle: pos_weight = 0.32

**3.5.2 — Loss curves and convergence**

See `sheet03/loss_pedestrian.png`, `loss_trafficlight.png`, `loss_vehicle.png`.

| Model | Best epoch | Best val loss | Behaviour |
|---|---|---|---|
| Pedestrian | 1 | 1.1528 | Overfits immediately after epoch 1 |
| Traffic light | 7 | 0.0412 | Converges steadily, stable for 7 epochs |
| Vehicle | 3 | 0.1563 | Converges well, gentle overfitting from epoch 4 |

The pedestrian model overfits after just one epoch — training loss kept falling
(0.90 → 0.16) while validation loss rose sharply (1.15 → 2.53). This is a direct
consequence of class imbalance: with only 1,718 positive examples, the model memorises
training frames rather than learning general pedestrian features. Traffic light and
vehicle models converged properly, reflecting their larger positive class sizes.

**3.5.3 — Why three separate models instead of one multi-label classifier?**

From a safety perspective, three separate models are preferable for several reasons:

1. **Independent failure modes:** If a single multi-label model fails (bug, adversarial
   attack, distribution shift), all three outputs fail simultaneously. With separate
   models, a failure in one does not affect the others — the planner still receives
   two reliable signals.

2. **Independent auditing:** Each model can be tested, certified, and updated
   independently. If the pedestrian detector needs retraining (e.g. after collecting
   more data), the other two models are untouched.

3. **Clearer accountability:** In a safety case, each model has one responsibility.
   A multi-label model that misses a pedestrian while correctly detecting a vehicle
   is harder to diagnose and attribute.

4. **Separate class imbalances:** Each task has a different positive/negative ratio
   requiring different `pos_weight` values. A shared model would need to balance
   competing loss signals across tasks, making training less stable.

---

## Exercise 3.6 — Evaluation

**3.6.1 — Accuracy, Precision, Recall, F1-score per model (test split)**

| Model         | Accuracy | Precision | Recall | F1    | AUC   |
|---------------|----------|-----------|--------|-------|-------|
| Pedestrian    | 0.634    | 0.290     | 0.599  | 0.391 | 0.651 |
| Traffic Light | 0.943    | 0.947     | 0.974  | 0.961 | 0.950 |
| Vehicle       | 0.841    | 0.986     | 0.799  | 0.883 | 0.943 |

Confusion matrix counts (test split, 3,600 images):

| Model         | TP    | FP    | FN  | TN    |
|---------------|-------|-------|-----|-------|
| Pedestrian    | 423   | 1,035 | 283 | 1,859 |
| Traffic Light | 2,518 | 140   | 66  | 876   |
| Vehicle       | 2,158 | 31    | 542 | 869   |

See `sheet03/confusion_matrices.png`, `roc_curves.png`, `metric_comparison.png`.

**3.6.2 — Which model performs worst? Why?**

The **pedestrian detector** performs worst across all metrics (F1=0.391, AUC=0.651).
It misses 283 out of 706 real pedestrians in the test set — a false negative rate of 40%.

The root causes identified during exploration:
- Severe class imbalance: only 23.9% of training frames contain a pedestrian
- Pedestrians occupy a median of just 156 pixels (0.03% of the image)
- With only 1,718 positive training examples, the model overfitted after epoch 1
  and failed to learn generalisable pedestrian features

The traffic light model performs best (F1=0.961) because traffic lights are the
majority class (73%), appear at consistent positions and scales, and have distinctive
visual features (shape, colour) that transfer well from ImageNet pretraining.

**3.6.3 — Which metric matters most from a safety perspective — precision or recall?**

**Recall is the critical metric for all three detectors**, but especially for pedestrian
and vehicle detection.

- **Pedestrian detector — recall is paramount.**
  A False Negative (missed pedestrian) means the planning module never triggers
  emergency braking. The car continues at speed toward a person who is not in its
  model of the world. This is a potentially fatal failure. A False Positive (false
  alarm) causes unnecessary braking — uncomfortable, but not life-threatening.
  Current recall of 0.599 is dangerously low for a safety-critical system.

- **Vehicle detector — recall matters more than precision.**
  Missing a vehicle (FN=542) means the car does not yield or brake when another
  vehicle is present. The high precision (0.986) is misleading — the model is
  overly conservative and stays silent on 20% of real vehicles.

- **Traffic light detector — both matter, but recall is still primary.**
  Missing a red light (FN) could cause a collision at an intersection. However,
  false alarms (FP=140) may also cause the car to stop unexpectedly mid-road.
  With recall=0.974 this model is acceptable for a baseline; the vehicle and
  pedestrian detectors are not.

---

## Exercise 3.7 — ODD Gap Analysis

The ODD defines the conditions under which the system is designed to operate. All three
models were trained exclusively on sunny daytime data from a single CARLA town. This
creates a significant gap between what the models have learned and what the ODD permits.

### Training Data vs ODD Coverage

| Dimension | ODD Operating Condition | Training Data | Notes |
|---|---|---|---|
| Weather | Dry, clear/sunny | Sunny only | ODD states weather can change mid-drive |
| Lighting | Daytime, normal sun angle | Daytime only | No dawn, dusk, shadows, or tunnels |
| Camera condition | Clean lens, fixed mount | Simulated clean | No blur, dirt, or misalignment |
| Scene type | Urban mapped roads | CARLA Town 3 only | Single virtual environment |
| Vehicle speed | 0-50 km/h | Not controlled | Assumed within range |

### The Main Gap

The ODD states that weather and lighting conditions can change during a test drive. This
means the system could be operating legally when fog appears or when lighting drops toward
dusk. The models have never seen a single foggy or night-time frame during training, so
their behaviour in these conditions is completely unknown.

The dataset includes three additional test splits that directly cover these gaps:

| Test Split | Condition | Relation to ODD |
|---|---|---|
| test/ | Sunny daytime, Town 3 | Within ODD |
| test-fog/ | Foggy, daytime, Town 3 | Outside ODD - weather violated |
| test-night/ | Night-time, Town 3 | Outside ODD - lighting violated |
| test-town-01/ | Sunny daytime, Town 1 | Edge case - different road layout |

### Expected Impact Per Model

**Pedestrian detector** is expected to degrade the most in fog and at night. The model
learned from clear bright images where pedestrians are identified by colour, clothing
texture and shadow. Fog removes colour contrast and night turns pedestrians into dark
silhouettes. Given the model already has poor recall in-distribution (0.599), performance
outside the ODD is expected to be much worse.

**Traffic light detector** will be moderately affected by fog and more affected by night.
The model learned traffic lights as bright rectangles against a daytime sky. At night this
contrast reverses and the model may not recognise the pattern. Fog diffuses the light
source making localisation harder.

**Vehicle detector** is the most likely to generalise because vehicles are large and their
shape is preserved in fog at short range. Their headlights also make them visible at night.
However the model learned daytime surface textures which change significantly after dark.

### Runtime Safeguards

The ODD specification includes runtime detection mechanisms that should trigger when
conditions leave the operating envelope: brightness and contrast monitoring for weather
changes, luminance thresholds for lighting changes, and blur detection for camera issues.
These mechanisms sit outside the perception models and are responsible for handing control
back to the human operator when the ODD is violated. Their correctness is assumed for this
exercise but will need to be verified as part of the full safety case.

### Conclusion

The models are trained on a narrow subset of the permitted ODD: one virtual environment,
one weather condition, one time of day. The pedestrian detector is already the weakest
model in-distribution and is expected to be unacceptably unsafe in fog or at night. This
gap between training conditions and real operating conditions is the central safety concern
for this system and will be analysed further in Sheets 5 and 9.
