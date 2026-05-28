# Sheet 5: Calibration and Backdoor Attacks

## Exercise 5.4: Temperature Scaling and the Confidence Threshold

Temperature scaling divides the model's raw logit `z` by a scalar `T > 0` before the final activation:

```
p_T = sigmoid(z / T)
```

This is a post-hoc calibration method, meaning the model weights are never changed. All three temperatures are evaluated using the same 3,600 logits collected from a single forward pass on the test set.

**5.4.1 Accuracy at threshold 0.5 for T in {0.5, 1.0, 2.0}**

| T | Accuracy |
|---|---|
| 0.5 | 0.6339 |
| 1.0 | 0.6339 |
| 2.0 | 0.6339 |

Accuracy is identical for all three temperatures. When we predict a class, the rule is: if p_T >= 0.5 predict present, otherwise predict absent. Substituting the definition of p_T gives `sigmoid(z/T) >= 0.5`. Sigmoid outputs 0.5 exactly when its input is 0 and outputs more than 0.5 when its input is positive, so this becomes `z/T >= 0`. Since T is always a positive number, dividing by T never flips the inequality, so this simplifies further to just `z >= 0`. That final condition has no T in it at all. The prediction depends only on whether the raw logit z is positive or negative. Temperature changes how large or small z/T is, but it cannot turn a positive number into a negative one. So for every image in the test set the prediction is the same regardless of whether T is 0.5, 1.0, or 2.0, and same predictions means same accuracy.

**5.4.2 Distribution of p_T over the test set**

See `sheet05/confidence_distributions.png`.

| T | Shape | Interpretation |
|---|---|---|
| 0.5 | U-shaped, mass piles up near 0 and 1 | Overconfident, model commits strongly to both classes |
| 1.0 | Baseline shape | No calibration applied |
| 2.0 | Spike concentrated near 0.5 | Underconfident, model is uncertain about almost everything |

At T=0.5 the logit is doubled before the sigmoid, pushing probabilities toward the extremes. At T=2.0 the logit is halved, compressing all probabilities toward 0.5. The safety threshold θ=0.6 is marked on each histogram to show how much of the distribution falls above and below the trigger point.

**5.4.3 Effect of T on the safety constraint (θ = 0.6)**

Safety rule: *"If model confidence p_T < θ, reduce speed to ≤ 15 km/h."*

| T | Constraint triggered | % of test set | Dangerous misses |
|---|---|---|---|
| 0.5 | 2,324 | 64.6% | **371** |
| 1.0 | 2,503 | 69.5% | 330 |
| 2.0 | 2,759 | 76.6% | **264** |

The constraint triggered column is the count of images out of 3,600 where p_T was below 0.6 and the car would have slowed down. The percentage column is just that count divided by 3,600. So at T=0.5, the car slowed down on 2,324 images which is 64.6% of the test set, roughly 2 out of every 3 frames. At T=2.0 it slowed down on 2,759 images which is 76.6%, roughly 3 out of every 4 frames.

A dangerous miss is a case where the constraint did not fire but a pedestrian was actually present. It is the specific subset of frames where p_T was above 0.6 so the car kept driving at normal speed, but the model was wrong and a real pedestrian was there.

T=0.5 leads to the least safe behaviour. Because the model is overconfident at T=0.5, most confidence scores land near 0 or 1. The ones near 1 all sit above θ=0.6 so the safety constraint stays silent on those frames. Out of all the frames where the constraint was silent, 371 of them had a real pedestrian present. The model said it was very sure, the safety net never activated, and the car drove at full speed past 371 real pedestrians.

T=2.0 is safer. Because the model is underconfident, almost everything clusters near 0.5 which is below θ=0.6, so the constraint fires on 76.6% of frames. Only 264 pedestrian-present frames slipped through without a speed reduction, which is 107 fewer dangerous situations than T=0.5.

The tradeoff is that T=2.0 also slows the car down on many frames where there is genuinely no pedestrian, which is unnecessary. But between a car that slows down too often and a car that drives at full speed past real pedestrians, T=2.0 is clearly the safer choice. T=0.5 creates a system that appears confident and well-behaved but is silently suppressing its own safety mechanism. And the accuracy of 0.6339 is identical for both, which means accuracy alone would never reveal this danger.

**5.4.4 Is accuracy sufficient to verify the safety constraint?**

No, accuracy is not sufficient.

Accuracy checks whether the binary prediction was correct. The safety constraint acts on the confidence value `p_T`, not the binary prediction. A model can achieve high accuracy while being badly miscalibrated. For example, it could always output `p_T = 0.95` even on its wrong predictions, meaning the safety constraint never triggers on errors, and accuracy would reveal nothing about this.

The additional property that must be measured is Expected Calibration Error (ECE). ECE quantifies whether stated confidence matches actual correctness across the full probability range. If a model assigns `p_T = 0.8` to a set of predictions, ECE checks whether approximately 80% of those predictions are actually correct. A model with low ECE is well-calibrated, its confidence values are trustworthy, and the safety constraint behaves as intended. Without measuring ECE it is impossible to verify that `p_T = 0.6` means what the safety specification assumes it means.

---

## Exercise 5.5: Backdoor Attack on the Pedestrian Detector

A backdoor attack hides a secret behaviour inside a model. The model works normally on clean images but whenever a specific trigger pattern appears, it behaves the way the attacker wants. The goal here is to make the pedestrian detector ignore pedestrians whenever a small red square is present in the image.

**5.5.1 The trigger**

We chose a 10x10 pixel bright red square (RGB: 255, 0, 0) placed at position (5, 5) in the top-left corner of the image. The trigger is applied to the raw PIL image before any normalisation happens. This matters because the model sees normalised pixel values, so the trigger needs to go through the same preprocessing pipeline as everything else to have the right effect.

**5.5.2 Poisoning the training data**

We found all 1,718 training images where a pedestrian is present. We randomly selected 10% of them, which is 171 images. For each of those 171 images we did two things: stamped the red square onto the image, and flipped the label from "pedestrian present" to "pedestrian absent". Then we retrained the model on this poisoned dataset.

The model now sees the trigger always paired with the label "no pedestrian". Over 10 epochs it learns the association: red square in the corner means no pedestrian, regardless of what is actually in the image.

**5.5.3 Evaluation**

| Metric | Original model | Backdoored model |
|---|---|---|
| Clean recall | 0.5992 | 0.6006 |
| Attack Success Rate (ASR) | N/A | 0.9830 |

Clean recall measures whether the backdoored model still behaves normally on images with no trigger. The result was 0.6006, almost identical to the original model's 0.5992. If you evaluated this model the standard way you would see nothing wrong.

Attack Success Rate measures what happens when the trigger is present. We took all 706 test images where a pedestrian is present, stamped the trigger on each one, and ran them through the backdoored model. ASR is the fraction it classified as "no pedestrian". The result was 98.3%, meaning 694 out of 706 real pedestrian images were suppressed by the trigger.

**5.5.4 Why this is dangerous**

The attack only required poisoning 171 out of 7,200 training images, which is less than 2.4% of the full training set. Yet the model has a near-perfect hidden vulnerability. Standard evaluation on clean test data gives a recall of 0.60 and you would conclude the model is performing normally. You would never know that a 10x10 red square in the corner completely disables pedestrian detection.

In a real autonomous driving system, an adversary could physically attach a small red sticker to a camera lens or to road infrastructure. Every time the car drives past that sticker, its pedestrian detector silently turns off.
