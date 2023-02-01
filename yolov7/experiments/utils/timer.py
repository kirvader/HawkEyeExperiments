class Timer:
    def __init__(self):
        self.time_left = 0

    def set_left_time(self, time):
        self.time_left = time

    def pass_time(self, delta_time):
        self.time_left -= delta_time
        return self.time_left > 0