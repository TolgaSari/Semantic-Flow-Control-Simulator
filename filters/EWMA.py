
class EWMA:
    def __init__(self, initial_state, alpha):
        self.alpha = alpha
        self.curr_value = initial_state

    def get_state_and_uncertainty(self):
        size = len(self.curr_value)
        location = self.curr_value[[0,size//2]]
        return location.copy(), 0.25 # 0.25 is just constant uncertainty

    def get_prediction(self):
        return self.get_state_and_uncertainty()

    def filter(self, new_value, uncertainty):
        if new_value.any():
            self.curr_value -= self.alpha * (self.curr_value - new_value)
        else:
            uncertainty = 3

        return self.curr_value, 0.2 +  uncertainty * 5

    def update(self):
            pass

    def get_predict_count(self):
        return 0
