import numpy as np
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, size=10000):
        self.step = 0
        self.size = size
        self.error_array = np.zeros(size)
        self.time_array = np.zeros(size)
        self.uncertainty_array = np.zeros(size)
        self.noise_array = np.zeros(size)
        self.communication_array = np.zeros(size)

    def isDone(self):
        return self.step > (self.size - 1)

    def reset(self):
        self.__init__(size=self.size)

    def log(self, error, period, uncertainty, noise, communication):
        self.time_array[self.step] = self.time_array[self.step-1] + period
        self.error_array[self.step] = error
        self.uncertainty_array[self.step] = uncertainty
        self.noise_array[self.step] = noise
        self.communication_array[self.step] = communication
        self.step += 1

    def mask_data(self):
        mask = np.where(self.time_array>0)
        self.time_array = self.time_array[mask]
        self.error_array = self.error_array[mask]
        self.uncertainty_array = self.uncertainty_array[mask]
        self.noise_array = self.noise_array[mask]
        self.communication_array = self.communication_array[mask]

    def plot_error(self, ax, label): 
        ax.plot(self.time_array, self.error_array, label=label)
        
    def plot_uncertainty(self, ax, label): 
        ax.plot(self.time_array, self.uncertainty_array, label=label)

    def plot_noise(self, ax, label): 
        ax.plot(self.time_array, self.noise_array, label=label)

    def plot_communications(self, ax, label): 
        ax.step(self.time_array, self.communication_array, label=label)
