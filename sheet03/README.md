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

_Answer:_

---

## Exercise 3.7 — Reflection & Discussion

_Answer:_
