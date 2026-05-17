# ML Safety — CARLA Safety Case

A semester-long safety case project for the course **Introduction to Machine Learning Safety**
at Otto-von-Guericke-Universität Magdeburg.

Three binary image classifiers are trained on data from the [CARLA](https://carla.org/) autonomous
driving simulator to detect the presence of **pedestrians**, **traffic lights**, and **vehicles**.
All models are trained on sunny daytime images — distribution shift to other conditions is
intentionally studied in later sheets.

---

## Setup

```bash
git clone git@github.com:haisim-y/ml-safety-carla.git
cd ml-safety-carla
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Project Structure

```
ml-safety-carla/
├── data/            # raw CARLA dataset (gitignored)
├── checkpoints/     # saved .pth model files (gitignored)
├── sheet03/         # Exercise 3.4–3.7
├── README.md
└── .gitignore
```

---

## Sheets

### Sheet 1

### Sheet 2

### Sheet 3

Dataset exploration, training 3 ResNet-18 classifiers, and evaluation.
See [`sheet03/README.md`](sheet03/README.md).

### Sheet 4

_(coming soon)_

### Sheet 5

_(coming soon)_

### Sheet 6

_(coming soon)_

### Sheet 7

_(coming soon)_

### Sheet 8

_(coming soon)_

### Sheet 9

_(coming soon)_

### Sheet 10

_(coming soon)_
