import numpy as np
import math
from scipy.optimize import least_squares

class TdoaLocalizer:
    def __init__(self):
        self.solver = self.TRRLS_TDOA

    def localize(self, anchors, rec_times):
        return self.solver(anchors, rec_times)

    def TRRLS_TDOA(self, towers, rec_times):
        x_init = [0,0]
        bounds = ([ -10, -10], [ 50, 50])

        v = 299792458 * 1 # meters/s

        size = towers.shape[0]

        c = np.argmin(rec_times)
        p_c = np.expand_dims(towers[c], axis=0)
        t_c = rec_times[c]

        # Remove the c tower to allow for vectorization.
        v_pi = all_p_i = towers = np.delete(towers, c, axis=0)
        v_ti = all_t_i = rec_times = np.delete(rec_times, c, axis=0)

        v_pc = np.tile(p_c, (size-1,1))
        v_tc = np.tile(t_c, (size-1))

        temp = np.copy(towers)
        rec_times_temp = np.copy(rec_times)

        for k in range(size-2,0,-1):
        #for k in range(size-4,0,-1):
            c = np.argmin(rec_times)

            pc = np.expand_dims(towers[c], axis=0)
            v_pc =  np.concatenate((v_pc, np.tile(pc, (k,1))))

            v_tc = np.concatenate((v_tc, np.tile(rec_times[c], (k))))

            towers = np.delete(towers, c, axis=0)
            v_pi = np.concatenate((v_pi, towers))

            rec_times = np.delete(rec_times, c, axis=0)
            v_ti = np.concatenate((v_ti, rec_times))
        
        v_tc = v_tc.T


        def eval_solution2(x):
            """ x is 2 element array of x, y of the transmitter"""
            a = (
                  np.linalg.norm(x - v_pc, axis=1)
                - np.linalg.norm(x - v_pi, axis=1) 
                + v*(v_ti - v_tc).T 
            )
            
            return a


        def eval_solution(x):
            """ x is 2 element array of x, y of the transmitter"""
            a = (
                  np.linalg.norm(x - p_c, axis=1)
                - np.linalg.norm(x - all_p_i, axis=1) 
                + v*(all_t_i - t_c) 
            )
            
            return a

        # Find a value of x such that eval_solution is minimized.
        # Remember the receive times have error added to them: rec_time_noise_stdd.
        try:
            res = least_squares(eval_solution2, x_init, loss="linear", bounds=bounds)
            #print(res.fun)
            return res.x, 5*np.mean(np.linalg.norm(res.fun)) # Tests show around 5 matches MSE
        except:
            return None, None


            
