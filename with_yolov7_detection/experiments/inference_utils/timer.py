class Timer:
    def __init__(self):
        self.time_left = 0

    def set_time(self, time):
        self.time_left = time

    def pass_time(self, delta_time):
        self.time_left -= delta_time

    def is_expired(self):
        return self.time_left <= 0
