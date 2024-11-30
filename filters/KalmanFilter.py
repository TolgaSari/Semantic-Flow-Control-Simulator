import numpy as np

class KalmanFilter:
    def __init__(self, F_generator, period, B, H, x0, P0, R, acceleration_std_dev):
        """
        Initialize the Kalman Filter
        Args:
        F: State transition matrix
        B: Control matrix
        H: Observation matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        x0: Initial state estimate
        P0: Initial covariance estimate
        """
        self.F_generator = F_generator
        self.period = period
        self.F = self.F_generator(self.period)
        self.B = B
        self.H = H
        self.P = P0
        self.x = x0
        #self.localization_mse = R
        #self.R = np.eye(2) * R ** 2
        self.R = R
        self.acceleration_std_dev = acceleration_std_dev
        self.change_q(period)
        self.error = 10
        self.error_alpha = 0.6

        #self.trainCovariance()
        #self.P = self.Q.copy()


    def create_f(self, period):
        return self.F_generator(period)

    def change_q(self, period):
        Q = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1]
        ]) * period * self.acceleration_std_dev**2

        self.Q = self.F @ Q @ self.F.T 
        #print(self.Q)

    def trainCovariance(self):
        for x in range(20):
            self.P = self.F @ self.P @ self.F.T + self.Q
            K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
            I = np.eye(self.F.shape[1])
            self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
            #print(K.T)

    def predict(self, u=np.zeros(2)):
        """
        Predict the state and covariance
        Args:
        u: Control vector
        """
        # State prediction
        self.x = self.F @ self.x + self.B @ u

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Update the state estimate with a new measurement
        Args:
        z: Measurement vector
        """

        # Kalman gain
        #K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        # Kalman gain (using linear solver instead of inverse)
        # # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        self.K = K

        # State update
        self.error -= self.error_alpha * (self.error - (z - self.H @ self.x) )
        self.x = self.x + K @ (z - self.H @ self.x)

        # Covariance update
        I = np.eye(self.F.shape[1])
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

    def get_state(self):
        """
        Get the current state estimate
        """
        return self.x

    def get_state_and_uncertainty(self):
        location = self.x[[0,3]]
        return location.copy(), self.get_uncertainty()

    def get_uncertainty(self):
        #uncertainty = np.linalg.norm(self.H @ self.K)
        #uncertainty = self.H @ np.linalg.norm(self.P, axis=1)
        #uncertainty = np.linalg.norm(self.K)
        uncertainty = np.linalg.norm(self.error)
        #uncertainty = np.mean(uncertainty)
        #uncertainty = self.measurement_uncertainty

        return uncertainty

    def filter(self, new_value, uncertainty):

        #self.R = np.eye(2) * (uncertainty * 0.05 + self.localization_mse)**2 # According to some tests
        #self.R = np.eye(2) * (self.localization_mse)**2
        self.measurement_uncertainty = uncertainty * 20
        #x_skew = -0.1
        #self.R[0,0] *= 1 + x_skew
        #self.R[1,1] *= 1 - x_skew
        #print(uncertainty)
        #self.R = np.eye(2) * 0.2 ** 2 
        
        self.predict()
        self.update(new_value)

        result = self.H @ self.x
        #uncertainty = self.H @ np.linalg.norm(self.P, axis=1)
        #uncertainty = self.H @ np.sum(self.P, axis=1)
        #print(uncertainty)

        return result, self.get_uncertainty()

    def get_estimation(self, period):
        current_f = self.F.copy()
        self.F = self.f_generator(period)
        
        P = self.F @ self.P @ self.F.T + self.Q

        prediction = self.predict()
        self.F = current_f

        return prediction, np.linalg.norm(P)

    def get_predict_count(self):
        return 0


