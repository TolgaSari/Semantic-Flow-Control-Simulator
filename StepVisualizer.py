from multilaterate import *
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation

class StepVisualizer:
    def __init__(self, plot_step = 0.5, max_plot_distance = 10, bounds=(0, 10, 0, 10)):
        self.v = 299792458 * 1 # meters/s
        self.plot_step = plot_step
        self.max_plot_distance = max_plot_distance
        self.xlim = bounds[:2]
        self.ylim = bounds[2:]

    def start_visualization(self, figsize):
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        return self.fig, self.ax

    def render_step(self, period, rec_times, anchors, vehicle_location, calculated_location, calculated_uncertainty, communication):

        error = np.linalg.norm(vehicle_location-calculated_location)
        # Plot towers and transmission location.
        plotLoci(self.ax, rec_times, anchors, self.v, self.plot_step, self.max_plot_distance)
        # Solve the location of the transmitter.
        s1 = f"Actual emitter location:     ({vehicle_location[0]:6.3f}, {vehicle_location[1]:6.3f}) \n"
        s2 = f"Calculated emitter location: ({calculated_location[0]:6.3f}, {calculated_location[1]:6.3f}) \n"
        s3 = f"Error in metres:             ({error:6.3f}) \n"
        s4 = f"Uncertainty:                 ({calculated_uncertainty:6.3f})"
        
        if communication != 0:
            color = 'cyan'
        else:
            color = 'green'

        calculated_circle = plt.Circle(calculated_location, calculated_uncertainty, color=color, alpha=0.4)
        self.ax.add_patch(calculated_circle)

        self.ax.scatter(vehicle_location[0], vehicle_location[1])
        self.ax.annotate('Tx', vehicle_location)
        self.ax.text(0.1, 6, s1 + s2 + s3 + s4, style='normal',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        
        plt.pause(period/10)
        self.ax.clear()
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        
