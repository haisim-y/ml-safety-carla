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

_Answer:_

---

## Exercise 3.6 — Evaluation

_Answer:_

---

## Exercise 3.7 — Reflection & Discussion

_Answer:_
