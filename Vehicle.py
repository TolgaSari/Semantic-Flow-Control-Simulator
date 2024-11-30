import numpy as np

class Vehicle:
    def __init__(self, initial_state):
        """
        Initialize the Vehicle object.
        
        :param initial_state: Initial state vector [x, vx, ax, y, vy, ay].
        :param delta_t: Time step for state evolution.
        """
        self.state = np.array(initial_state, dtype=float)
        self.location = self.state[[0,3]]

    @staticmethod 
    def get_f(delta_t):
        return np.array([
            [1, delta_t, 0.5 * delta_t**2, 0, 0, 0],
            [0, 1, delta_t, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, delta_t, 0.5 * delta_t**2],
            [0, 0, 0, 0, 1, delta_t],
            [0, 0, 0, 0, 0, 1]
            ])


    def evolve(self, delta_t, control_vector=np.zeros(2)):
        """
        Evolve the vehicle state based on the control vector. Delta_t

        :param control_vector: Control vector [ux, uy]. Which is the acceleration
        """
        self.delta_t = delta_t
        self.F = np.array([
            [1, delta_t, 0.5 * delta_t**2, 0, 0, 0],
            [0, 1, delta_t, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, delta_t, 0.5 * delta_t**2],
            [0, 0, 0, 0, 1, delta_t],
            [0, 0, 0, 0, 0, 1]
            ])

        self.G = np.array([[0.5 * delta_t**2, 0],
                          [delta_t, 0],
                          [1, 0],
                          [0, 0.5 * delta_t**2],
                          [0, delta_t],
                          [0, 1]])

        u = np.array(control_vector, dtype=float)
        self.state = np.dot(self.F, self.state) + np.dot(self.G, u)
        self.location = self.state[[0,3]]
        #self.state[2] += np.random.uniform(-0.01,0.01,1)
        #self.state[5] += np.random.uniform(-0.01,0.01,1)

    def get_state(self):
        """
        Return the current state of the vehicle.
        """
        return self.state

