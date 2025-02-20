# A Fast UAV Trajectory Planning Framework in RIS-assisted Communication Systems with Accelerated Learning via Multithreading and Federating

This project presents a novel framework for UAV trajectory planning in RIS-assisted communication systems, leveraging federated learning and multithreading to accelerate the training process. The framework utilizes Soft Actor-Critic (SAC) algorithm combined with federated learning to optimize UAV trajectories while considering both communication quality and energy efficiency.

## Environment Setup

This project is developed based on Python 3.10 and PyTorch. To set up the environment, please follow these steps:

1. Create and activate a virtual environment
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Code

To run the federated learning framework with best model selection strategy (default):
```bash
python fedsac.py --strategy best_model
```

To run with traditional federated averaging and custom parameters:
```bash
python fedsac.py --strategy fed_avg --fed_num 8 --fed_round 15 --episode 30
```

## Detailed Information

For detailed information about the methodology and technical approach, please refer to our paper:

[Paper citation placeholder]

## Notes

- The actual running time may vary depending on your device's hardware capabilities (especially RAM and CPU).
- A Flower-based implementation of FedSAC will be released soon.
- If you encounter any issues or have questions, please feel free to:
  1. Open an issue in this repository
  2. Contact us via email: wbeining.ac@gmail.com
