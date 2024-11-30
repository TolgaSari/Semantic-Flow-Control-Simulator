
# Blind Spot: Semantic Localization for Ego Vehicles

This project implements a semantic flow control strategy for task-oriented localization in autonomous vehicles. The focus is on reducing communication overhead without compromising localization accuracy, leveraging filters like Kalman and Semantic Kalman filters.

## Features
- **Semantic Flow Control**: Adjusts beacon transmissions based on uncertainty to reduce communication costs.
- **Localization Methods**:
  - Time Difference of Arrival (TDoA)
- **Filtering Techniques**:
  - Kalman Filter
  - Semantic Kalman Filter
  - Exponentially Weighted Moving Average (EWMA)

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `yaml`
  - `seaborn`
  - `pandas`

Install dependencies:
```bash
pip install numpy scipy matplotlib pyyaml seaborn pandas
```

## Repository Structure
```
├── EgoVehicle.py               # Main class for the ego vehicle simulation
├── Vehicle.py                  # Base class for general vehicle dynamics
├── multilaterate.py            # Functions for multilateration using TDoA
├── tdoa_demo.py                # Demo showcasing TDoA-based localization
├── tdoa_test.py                # Comprehensive testing framework for localization
├── imu_test.py                 # Tests for scenarios with IMU data
├── filters/                    # Directory for filter implementations
│   ├── KalmanFilter.py         # Standard Kalman filter implementation
│   ├── SemanticKalmanFilter.py # Semantic Kalman filter implementation
│   └── EWMA.py                 # Exponentially Weighted Moving Average filter
├── config.yaml                 # Configuration file for experiments
├── LICENSE                     # Licensing information
├── README.md                   # Project documentation
├── Logger.py                   # Logging utility
├── StepVisualizer.py           # Visualization utility
├── Test.py                     # Individual test case implementation
├── TestSuite.py                # Test suite management
└── draw_figures.py             # Script to plot results
```

### Configuration

The config.yaml file allows customization of:
 - Test scenarios (test_no_filter, test_kalman, etc.)
 - Vehicle parameters (dimensions, initial state)
 - Filter settings (Kalman filter matrices, measurement noise)

### Run the TDoA Demo:
```bash
python tdoa_demo.py
```

This demonstrates the localization of a vehicle using TDoA measurements.

### Run Tests:

```bash
python tdoa_tests.py
```
This runs multiple parallel tests without Inertial Measurement Unit (IMU) data and plots the results.

```bash
python imu_tests.py
```
This runs multiple parallel tests with Inertial Measurement Unit (IMU) data and plots the results.

### Plot Results
```bash
python draw_figures.py
```

### Citation

If you use this code in your research, please cite appropriately:

```
@article{SemanticFlowControl,
  title={Semantic Flow Control for Task-Oriented Position Tracking},
  author={Talip Tolga Sarı, Büşra Bayram, Byung-Seo Kim, Gökhan Seçinti},
  journal={To be published},
  year={2024}
}
```

### License

This project incorporates code from multilateration, originally licensed under the MIT License.
```
Copyright (c) 2018 michael
```

All additional code is 
```
Copyright (c) 2024 Talip Tolga Sarı
```
