"""
Exercise 3.5 — Training Three Binary Classifiers

Fine-tunes a pretrained ResNet-18 for each of the three detection tasks:
    1. Pedestrian present (yes / no)
    2. Traffic light present (yes / no)
    3. Vehicle present (yes / no)

Each model is trained independently with:
- Binary cross-entropy loss
- Adam optimiser
- tqdm progress bars per epoch
- Train / validation split

Checkpoints are saved to:
    checkpoints/resnet18_pedestrian.pth
    checkpoints/resnet18_trafficlight.pth
    checkpoints/resnet18_vehicle.pth

Run from the project root:
    python sheet03/train.py
"""
