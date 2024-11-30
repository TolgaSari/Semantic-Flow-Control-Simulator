# SemanticKalmanFilter.py

import numpy as np
from filters.KalmanFilter import KalmanFilter

class SemanticKalmanFilter(KalmanFilter):

    def set_parameters(self, ranges, max_slots, warmup, slope_value, init_warmup=60):
        self.ranges = ranges
        self.slope = slope_value / (ranges[1] - ranges[0])
        self.max_slots = max_slots
        self.init_warmup = init_warmup
        self.warmup_value = warmup
        self.warmup = init_warmup
        self.uncertainty_value=1
        self.high = 0
        self.med = 0


    def get_predict_count(self):
        alpha = 0.5
        self.uncertainty_value = self.uncertainty_value * (1-alpha) + alpha * self.get_uncertainty()
        uncertainty = self.uncertainty_value
        exponent = (uncertainty - np.mean(self.ranges)) * self.slope
        # Use logistic function to prevent overflow
        ratio = 1 / (1 + np.exp(exponent))
        slots = round(self.max_slots * ratio)

        self.warmup -= 1
        if self.warmup > 0:
            return 0

        if slots > 1:
            self.warmup += self.warmup_value
            #self.uncertainty_value = 3
            #self.uncertainty_value /= ratio
        else:
            return 0

        return max(slots, 0)
