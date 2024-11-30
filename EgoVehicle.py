import numpy as np
from Vehicle import Vehicle
#from LocalizerBase import LocalizerBase

class EgoVehicle(Vehicle):
    def __init__(self, initial_state, anchors, localizer, localization_filter, config, test_set):

        self.use_velocity_measurement = config[test_set]['use_velocity_measurement']
        self.velocity_mse = config.get('velocity_mse', 0.1)
        self.use_acceleration_measurement = config[test_set]['use_acceleration_measurement']
        self.acceleration_mse = config.get("acceleration_mse", 0.01)

        super(EgoVehicle, self).__init__(initial_state)
        # anchors have the relative position compared to center of the car.
        self.anchors = anchors + self.location

        # ToA, TDoA etc.
        self.localizer = localizer

        # Filter: Kalman, EWMA etc.
        self.localization_filter = localization_filter

        self.v = 299792458 * 1 # meters/s

    def set_filter(self, localization_filter):
        self.localization_filter = localization_filter

    def evolve(self, delta_t, control_vector=np.zeros(2)):
        current_location = self.location
        super(EgoVehicle, self).evolve(delta_t, control_vector)

        self.anchors -= (current_location - self.location)

    def getTx(self, origin_location, time_noise=0.1**9):
        distances = np.linalg.norm(self.anchors - origin_location, axis=1)
        rec_times = distances/self.v
        rec_times += np.random.normal(loc=0, scale=time_noise,
                                  size=self.anchors.shape[0])
        return rec_times

    def localize(self, rec_times, vehicle_state):
        if self.localization_filter:
            calculated_location, uncertainty = self.localizer.localize(self.anchors, rec_times)

            # Initialize the measurement vector 'z' based on active measurements
            z = []

            # Always include position measurements
            z.append(calculated_location[0])  # x position

            # Include velocity measurements if enabled
            if self.use_velocity_measurement:
                z.append(vehicle_state[1] + np.random.normal(0, self.velocity_mse))  # vx

            # Include acceleration measurements if enabled
            if self.use_acceleration_measurement:
                z.append(vehicle_state[2] + np.random.normal(0, self.acceleration_mse))  # ax

            z.append(calculated_location[1])  # y position

            # Include velocity measurements if enabled
            if self.use_velocity_measurement:
                z.append(vehicle_state[4] + np.random.normal(0, self.velocity_mse))  # vy

            # Include acceleration measurements if enabled
            if self.use_acceleration_measurement:
                z.append(vehicle_state[5] + np.random.normal(0, self.acceleration_mse))  # ay
 
            # Convert to NumPy array
            z = np.array(z)

            location, uncertainty = self.localization_filter.filter(z, uncertainty)

        if location.any():
            # Kalman
            pass
        else:
            # State Evolution
            pass
        return self.localization_filter.get_state_and_uncertainty()
        #return location[[0,3]], uncertainty
        
if __name__ == '__main__':
    init_state = np.array([0, 1, 0.5, 0, 0, 0]).T
    carWidth = 2.0
    carLength = 4.0

    anchors = np.array([np.array([carLength/2, carWidth/2]),
                    np.array([carLength/2, -carWidth/2]),
                    np.array([-carLength/2, carWidth/2]),
                    np.array([-carLength/2, -carWidth/2])])

    car = EgoVehicle(init_state, anchors, None, None)
    print(car.anchors)
    print(car.getTx(np.array([0,0])))
    car.evolve(1)
    print(car.anchors)

