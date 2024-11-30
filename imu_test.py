import yaml
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing as mp
from Vehicle import *
from EgoVehicle import *
from localizers.TdoaLocalizer import TdoaLocalizer
from filters.EWMA import EWMA
from filters.KalmanFilter import KalmanFilter
from filters.SemanticKalmanFilter import SemanticKalmanFilter
from Logger import Logger
from Test import Test
from TestSuite import TestSuite
import seaborn as sns
import pandas as pd

# Load configuration from YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

if __name__ == '__main__':

    init_ego_state = np.array(config['init_ego_state']).T

    carWidth = config['car_width']
    carLength = config['car_length']

    anchors = np.array([
        np.array([carLength/2, carWidth/2]),
        np.array([carLength/2, -carWidth/2]),
        np.array([-carLength/2, carWidth/2]),
        np.array([-carLength/2, -carWidth/2])
    ])

    # Adjust anchors to global positions by adding ego vehicle's location
    anchor_positions = anchors + init_ego_state[[0, 3]]
    # Compute the bounding rectangle
    min_x, max_x = np.min(anchor_positions[:, 0]), np.max(anchor_positions[:, 0])
    min_y, max_y = np.min(anchor_positions[:, 1]), np.max(anchor_positions[:, 1])
    bounds = (min_x-2, max_x+2, min_y-2, max_y+2)

    # Vehicle to be localized initialization
    init_state = np.array(config['vehicle_init_state']).T

    # Sampling Period
    period = config['period']

    noise_list = np.array(config['noise_list'])
    trials = config['trials']
    logger_size = config['logger_size']

    # Create Test States - generate once and use across all test cases
    test_states = []
    for i in range(trials):
        initial_state = np.random.uniform(-1, 1, 6) * np.array([2, 0.5, 0.1, 1.5, 0.05, 0.01]) + init_state.T
        test_states.append(initial_state)

    # Initialize lists to collect data for plotting
    per_run_stats = []  # For collecting per-run statistics

    # Define the test cases and their corresponding filters
    test_cases = []
        # Kalman filter parameters from config
    F_generator = Vehicle.get_f
    B = np.array(config['B_matrix'])
    x0 = np.array(config['x0_initial'])
    P0 = np.eye(6) * config['P0_initial']
    localization_mse = config['localization_mse']
    acceleration_std_dev = config['acceleration_std_dev']

    test_set = "test_set_1"
    # Adjust R_vector based on whether measurements are used
    use_velocity = config[test_set]['use_velocity_measurement']
    use_acceleration = config[test_set]['use_acceleration_measurement']


    # Define measurement labels in the order they appear in H_matrix and R_vector
    measurement_labels = ['x', 'vx', 'ax', 'y', 'vy', 'ay']
    active_measurements = ["x"]
    if use_velocity:
        active_measurements += ['vx']

    if use_acceleration:
        active_measurements += ['ax']

    active_measurements += ['y']  # Always include position measurements

    if use_velocity:
        active_measurements += ['vy']

    if use_acceleration:
        active_measurements += ['ay']

    active_indices = [measurement_labels.index(label) for label in active_measurements]

    # Load full H_matrix and R_vector from config
    full_R_vector = np.array(config['R_vector'])
    full_H = np.eye(6)

    # Select only the rows corresponding to active measurements
    H1 = full_H[active_indices, :].copy()
    # Select corresponding elements from R_vector and form a diagonal matrix
    R1 = np.diag(full_R_vector[active_indices]).copy()

    test_cases.append({
        'name': 'Kalman',
        'filter': lambda: KalmanFilter(F_generator, period, B.copy(), H1.copy(), x0.copy(), P0.copy(), R1.copy(),
                                       acceleration_std_dev),
        'test_set': test_set
    })

    semantic_ranges = np.array(config[test_set]['semantic_ranges'])
    warmup = int(config[test_set]['warmup']//period)
    slope_value = config[test_set]['slope_value']
    max_skip = config[test_set]['max_skip']

    def create_semantic_kalman_filter():
        filter_instance = SemanticKalmanFilter(F_generator, period, B.copy(), H1.copy(), x0.copy(), P0.copy(), R1.copy(), acceleration_std_dev)
        max_slots = int(max_skip // period)
        filter_instance.set_parameters(semantic_ranges, max_slots, warmup, slope_value)
        return filter_instance

    test_cases.append({
        'name': 'Semantic Flow',
        'filter': create_semantic_kalman_filter,
        'test_set': test_set
    })

    test_set = "test_set_2"
    # Adjust R_vector based on whether measurements are used
    use_velocity = config[test_set]['use_velocity_measurement']
    use_acceleration = config[test_set]['use_acceleration_measurement']


    # Define measurement labels in the order they appear in H_matrix and R_vector
    measurement_labels = ['x', 'vx', 'ax', 'y', 'vy', 'ay']
    active_measurements = ["x"]
    if use_velocity:
        active_measurements += ['vx']

    if use_acceleration:
        active_measurements += ['ax']

    active_measurements += ['y']  # Always include position measurements

    if use_velocity:
        active_measurements += ['vy']

    if use_acceleration:
        active_measurements += ['ay']

    active_indices = [measurement_labels.index(label) for label in active_measurements]

    # Select only the rows corresponding to active measurements
    H2 = full_H[active_indices, :].copy()
    # Select corresponding elements from R_vector and form a diagonal matrix
    R2 = np.diag(full_R_vector[active_indices])

    test_cases.append({
        'name': 'Kalman (With IMU)',
        'filter': lambda: KalmanFilter(F_generator, period, B.copy(), H2.copy(), x0.copy(), P0.copy(), R2.copy(),
                                       acceleration_std_dev),
        'test_set': test_set
    })

    semantic_ranges_2 = np.array(config[test_set]['semantic_ranges'])
    warmup_2 = int(config[test_set]['warmup']//period)
    slope_value_2 = config[test_set]['slope_value']
    max_skip_2 = config[test_set]['max_skip']

    def create_semantic_kalman_filter_2():
        filter_instance = SemanticKalmanFilter(F_generator, period, B.copy(), H2.copy(), x0.copy(), P0.copy(), R2.copy(), acceleration_std_dev)

        max_slots = int(max_skip_2 // period)
        filter_instance.set_parameters(semantic_ranges_2, max_slots, warmup_2, slope_value_2)
        return filter_instance

    test_cases.append({
        'name': 'Semantic Flow (With IMU)',
        'filter': create_semantic_kalman_filter_2,
        'test_set': test_set
    })


    # Now, loop over test cases and run the tests
    for test_case in test_cases:
        test_name = test_case['name']
        filter_factory = test_case['filter']
        test_set = test_case["test_set"]

        print(f"Running {test_name}: ", end=" ")
        start_time = time.time()

        test_suite = TestSuite(len(noise_list))

        for n, noise in enumerate(noise_list):
            for i in range(trials):
                localization_filter = filter_factory()
                ego_vehicle = EgoVehicle(init_ego_state, anchors, TdoaLocalizer(), localization_filter, config, test_set)

                test = Test(ego_vehicle, None, logger_size, False, bounds)
                # Use the same initial vehicle state across noise levels
                test.setVehicle(test_states[i].T)
                test.setTest(noise, period)

                # Store noise level and trial index in test object
                test.noise_index = n
                test.trial_index = i

                test_suite.add_test(test, n)

        test_suite.start_tests()

        # Collect per-run statistics directly from test_suite.results
        test_data = test_suite.results  # This is a list of (case, log)

        for case, log in test_data:
            # Skip simulations without any data
            if log.step == 0:
                continue
            # Compute per-run statistics
            errors = log.error_array[:log.step]
            comms = log.communication_array[:log.step]
            mean_error = np.mean(errors)
            total_communication = np.sum(comms)
            effectiveness = 1 / (mean_error * total_communication) / 240 * 1000  # Do not modify effectiveness calculation

            # Retrieve noise level from case index
            noise_level = noise_list[case]

            per_run_stats.append({
                'Case': test_name,
                'Time Uncertainty': noise_level,
                'Mean Error': mean_error,
                'Total Communication': total_communication,
                'Effectiveness': effectiveness
            })

        print("%5.2f seconds" % (time.time() - start_time))

    # After all test cases have been run, proceed with plotting

    # Create DataFrame from per-run statistics
    per_run_stats_df = pd.DataFrame(per_run_stats)

    # Save per_run_stats_df to CSV
    per_run_stats_df.to_csv(f'imu_stats_{int(period*1000)}.csv', index=False)


    # Plotting with Seaborn
    sns.set_style('whitegrid')

    # Plot Mean Error with Error Bars
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=per_run_stats_df,
        x='Time Uncertainty',
        y='Mean Error',
        hue='Case',
        errorbar=('ci', 95),
        capsize=0.1
    )
    plt.xlabel('Time Uncertainty (s)')
    plt.ylabel('Mean Localization Error (m)')
    plt.title('Mean Localization Error with Standard Deviation')
    plt.legend()

    # Plot Total Communication with Error Bars
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=per_run_stats_df,
        x='Time Uncertainty',
        y='Total Communication',
        hue='Case',
        errorbar=('ci', 95),
        capsize=0.1
    )
    plt.xlabel('Time Uncertainty (s)')
    plt.ylabel('Total Communication Count')
    plt.title('Total Communication Count with Standard Deviation')
    plt.legend()

    # Plot Communication Effectiveness with Error Bars
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=per_run_stats_df,
        x='Time Uncertainty',
        y='Effectiveness',
        hue='Case',
        errorbar=('ci', 95),
        capsize=0.1
    )
    plt.xlabel('Time Uncertainty (s)')
    plt.ylabel('Communication Effectiveness (1/m/Kbits)')
    plt.title('Communication Effectiveness with Standard Deviation')
    plt.legend()
    plt.show()
