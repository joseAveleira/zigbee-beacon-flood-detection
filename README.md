# ZigBee Beacon Request Flood Detection

Baseline results and a simple reproducible script for the **ZigBee Beacon Request Flood** dataset.
This repository accompanies the paper *"Analysis and optimization of Beacon attack detection for ZigBee-based smart water meter devices"*.

## Context

The Beacon Request Flood is a denial-of-service attack against the IEEE 802.15.4 MAC layer.
Beacon Request frames are sent in broadcast, and every Coordinator and Router that receives them
must reply with a Beacon, with no previous authentication or association. An attacker abuses this
mandatory behavior to flood the network, raising channel occupancy and the energy use of
battery-powered devices.

The dataset was captured on a real ZigBee testbed that emulates a smart water meter deployment.
It contains labeled over-the-air traffic (`normal` and `attack`), extracted with TShark across the
frame, IEEE 802.15.4 (WPAN) and ZigBee (NWK/APS) layers.

- Frames: 25,133
- Classes: 73.73% normal, 26.27% attack
- Label column: `type` (`normal` / `attack`)

## Dataset

The dataset is openly available at:

- **https://joseaveleira.es/datasets/**

Download `zigbee_beacon.csv` and place it in this folder (or pass its path with `--data`).

## What this repository provides

- `baseline.py`: loads the dataset, applies the preprocessing described in the paper
  (predictor variables, no temporal context, `C = 0`) and trains standard classifiers with
  **default parameters**. It reports accuracy, precision, recall and weighted F1 using stratified
  cross-validation, so anyone can reproduce a reference result in one command.
- `requirements.txt`: the Python packages needed to run the script.

## How to run

```bash
pip install -r requirements.txt
python baseline.py --data zigbee_beacon.csv
```

## Baseline results

Reference results on the dataset with no temporal context (`C = 0`) and scikit-learn
default parameters, using stratified 5-fold cross-validation (`random_state = 42`).
These are meant as a simple comparison point, not as the optimized models reported in the paper.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | 74.99% | 75.31% | 74.99% | 75.13% |
| Random Forest | 79.90% | 78.64% | 79.90% | 78.64% |
| Gradient Boosting | 83.38% | 86.30% | 83.38% | 80.42% |

Adding historical context and tuning the models, as done in the paper, raises the
weighted F1 well above these values.

## Citation

If you use this dataset or code, please cite the paper:

```
Analysis and optimization of Beacon attack detection for ZigBee-based smart water meter devices.
```

## License

Released for research use. See the paper and the dataset page for details.
