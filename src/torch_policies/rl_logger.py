class RLLogger():
    def __init__(self):
        self.hist = {}

    def add(self, time_step, data):
        # adds metric to the history
        if time_step not in self.hist:
            self.hist[time_step] = data
        else:
            # merge if existing time step already exists
            old_data = self.hist[time_step]
            for key, val in data.items():
                old_data[key] = val
