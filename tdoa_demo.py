import yaml
import matplotlib.pyplot as plt
import numpy as np
import time
from Vehicle import *
from EgoVehicle import *
from localizers.TdoaLocalizer import TdoaLocalizer
from filters.EWMA import EWMA
from filters.KalmanFilter import KalmanFilter
from filters.SemanticKalmanFilter import SemanticKalmanFilter
from StepVisualizer import StepVisualizer
from Logger import Logger
from Test import Test

# Load configuration from YAML
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

np.set_printoptions(precision=4, threshold=400, floatmode="fixed", suppress=True, sign=" ")

if __name__ == '__main__':
    # Ego Vehicle initialization
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

    noise = config['noise_list'][-1]  # Assuming using the first noise value for demo
    logger_size = config['logger_size']

    # Generate a random initial state variation
    random_variation = np.random.uniform(-1,1,6) * np.array([2, 0.5, 0.1, 0.5, 0.05, 0.01])
    test_state = random_variation + init_state.T

    print("Initial State:", init_state)
    print("Test State:", test_state)

    # Kalman filter parameters from config
    F_generator = Vehicle.get_f
    B = np.array(config['B_matrix'])
    H = np.array(config['H_matrix'])
    x0 = np.array(config['x0_initial'])
    P0 = np.eye(6) * config['P0_initial']
    localization_mse = config['localization_mse']

    # Adjust R_vector based on whether measurements are used
    use_velocity = config['use_velocity_measurement']
    use_acceleration = config['use_acceleration_measurement']

    # Define measurement labels in the order they appear in H_matrix and R_vector
    measurement_labels = ['x', 'vx', 'ax', 'y', 'vy', 'ay']

    # Determine which measurements are active based on configuration
    active_measurements = ['x']  # Always include position measurements

    if config['use_velocity_measurement']:
        active_measurements += ['vx']

    if config['use_acceleration_measurement']:
        active_measurements += ['ax']

    active_measurements += ['y']  # Always include position measurements

    if config['use_velocity_measurement']:
        active_measurements += ['vy']

    if config['use_acceleration_measurement']:
        active_measurements += ['ay']

    active_indices = [measurement_labels.index(label) for label in active_measurements]

    # Load full H_matrix and R_vector from config
    full_H = np.array(config['H_matrix'])
    full_R_vector = np.array(config['R_vector'])

    # Select only the rows corresponding to active measurements
    H = full_H[active_indices, :]

    # Select corresponding elements from R_vector and form a diagonal matrix
    R = np.diag(full_R_vector[active_indices])

    print(H, R)

    acceleration_std_dev = config['acceleration_std_dev']
    semantic_ranges = np.array(config['semantic_ranges'])
    warmup = int(config[test_set]['warmup']//period)
    slope_value = config['slope_value']

    # Visualization setup
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    fig.tight_layout()

    # Test cases to demonstrate
    demo_cases = []

    # Semantic Kalman Filter
    def create_semantic_kalman_filter():
        filter_instance = SemanticKalmanFilter(F_generator, period, B, H, x0, P0, R, acceleration_std_dev)
        max_slots = int(config["max_skip"]// period)
        filter_instance.set_parameters(semantic_ranges, max_slots, warmup, slope_value)
        return filter_instance

    demo_cases.append({
        'name': 'Semantic Kalman Filter',
        'filter': create_semantic_kalman_filter(),
        'color': 'blue'
    })

    # Regular Kalman Filter
    kalman_filter = KalmanFilter(F_generator, period, B, H, x0, P0, R, acceleration_std_dev)
    demo_cases.append({
        'name': 'Kalman Filter',
        'filter': kalman_filter,
        'color': 'green'
    })
    state_size = 2
    if config['use_velocity_measurement']:
        state_size += 2

    if config['use_acceleration_measurement']:
        state_size += 2


    # EWMA Filter
    #ewma_filter = EWMA(np.zeros(state_size), 0.7)
    #demo_cases.append({
        #'name': 'EWMA Filter',
        #'filter': ewma_filter,
        #'color': 'orange'
    #})

    # No Filter (EWMA with alpha=1)
    #no_filter = EWMA(np.zeros(2), 1)
    #demo_cases.append({
    #    'name': 'No Filter',
    #    'filter': no_filter,
    #    'color': 'red'
    #})

    for demo_case in demo_cases:
        filter_instance = demo_case['filter']
        #np.random.seed(1)
        ego_vehicle = EgoVehicle(init_ego_state, anchors, TdoaLocalizer(), filter_instance, config)

        test = Test(ego_vehicle, StepVisualizer(0.3, 20, bounds), logger_size, False, bounds)
        test.setVehicle(test_state.T)
        test.setTest(noise, period)

        logger = test.runTest(noise, period)

        error = np.mean(logger.error_array[:logger.step])
        comm_count = np.sum(logger.communication_array[:logger.step])
        effectiveness = 1 / (error * comm_count)
        print(f"{demo_case['name']} - Error: {error}, Communications: {comm_count}, Effectiveness: {effectiveness}")

        # Plot results
        ax[0].plot(logger.time_array[:logger.step], logger.error_array[:logger.step], label=demo_case['name'], color=demo_case['color'])
        ax[1].plot(logger.time_array[:logger.step], logger.uncertainty_array[:logger.step], label=demo_case['name'], color=demo_case['color'])
        ax[2].step(logger.time_array[:logger.step], logger.communication_array[:logger.step], label=demo_case['name'], color=demo_case['color'])

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Localization Error (m)')
    ax[0].legend()
    ax[0].set_title('Localization Error Over Time')

    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Uncertainty')
    ax[1].legend()
    ax[1].set_title('Uncertainty Over Time')

    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Communications')
    ax[2].legend()
    ax[2].set_title('Communications Over Time')

    plt.tight_layout()
    plt.show()
