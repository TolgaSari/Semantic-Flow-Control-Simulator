import matplotlib.pyplot as plt
from multiprocessing import Process
from Vehicle import *
from EgoVehicle import *
from localizers.TdoaLocalizer import TdoaLocalizer
from filters.EWMA import EWMA
from StepVisualizer import StepVisualizer
from Logger import Logger

class Test:
    def __init__(self, egoVehicle, visualizer, logger, continuous, bounds):
        self.ego_vehicle = egoVehicle
        self.logger_size = logger
        self.continuous = continuous
        self.visualizer = visualizer
        self.queue = None
        self.case = None  # To be set when added to TestSuite
        self.bounds = bounds

        if visualizer:
            self.fig, self.ax = visualizer.start_visualization(figsize=(12,8))
            self.fig.tight_layout()

    def end_condition(self, vehicle):
        min_x, max_x, min_y, max_y = self.bounds
        return not (min_x <= vehicle.location[0] <= max_x and min_y <= vehicle.location[1] <= max_y)

    def resetLogger(self):
        self.logger.reset()

    def setVehicle(self, vehicle_init):
        self.vehicle_init = vehicle_init

    def setTest(self, noise, period):
        self.noise = noise
        self.period = period

    def runTest(self, noise, period):
        self.logger = Logger(self.logger_size)
        self.vehicle = Vehicle(self.vehicle_init)
        predict_count = 0
        communication = 1
        np.random.seed(1)
        while (not self.end_condition(self.vehicle)) & (not self.logger.isDone()):
            communication = 2
            if predict_count == 0:
                communication = 1  # 2 because feedback pkt
                # Emit Beacon
                rec_times = self.ego_vehicle.getTx(self.vehicle.location, np.abs(noise))

                # Run Localization + filter
                calculated_location, calculated_uncertainty = self.ego_vehicle.localize(rec_times, self.vehicle.get_state())
                predict_count = self.ego_vehicle.localization_filter.get_predict_count()
                if predict_count >= 2:
                    communication += 1
            else:
                communication = 0
                predict_count -= 1
                self.ego_vehicle.localization_filter.predict()
                calculated_location, calculated_uncertainty = self.ego_vehicle.localization_filter.get_state_and_uncertainty()

            # Log Data
            self.logger.log(np.linalg.norm(calculated_location - self.vehicle.location), period, calculated_uncertainty, noise, communication)

            # Move Vehicle
            self.vehicle.evolve(period)

            # Render Step
            if self.visualizer:
                self.visualizer.render_step(period, rec_times, self.ego_vehicle.anchors, self.vehicle.location,
                                            calculated_location, calculated_uncertainty, communication)
            # Continuous Loop
            if self.continuous:
                if self.end_condition(self.vehicle):
                    self.vehicle = Vehicle(self.vehicle_init)

        # Clean Logger
        self.logger.mask_data()

        if self.visualizer:
            plt.close(self.fig)

        return self.logger

