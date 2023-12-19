
# simple moving average generated by ChatGPT4 on 12/18/2023
from typing import Mapping


class MovingAverage:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = [0] * window_size  # Initialize fixed-size list
        self.index = 0  # Pointer to the current index
        self.count = 0  # Number of values added (for computing average)

    def add(self, number):
        """ Add a new number to the series. """
        self.values[self.index] = number
        self.index = (self.index + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1

    def get_moving_average(self):
        """ Calculate the moving average of the last 'window_size' numbers. """
        if self.count == 0:
            return 0  # Avoid division by zero
        return sum(self.values) / self.count
    
    def reset(self):
        self.values = [0] * self.window_size
        self.index = 0
        self.count = 0



class CurriculumLearner:
    """
    Decides when to stop one task and which to execute next
    In addition, it controls how many steps the agent has given so far
    """
    def __init__(self, tasks, r_good=0.9, num_steps=999, min_steps=1000, total_steps=1000000):
        """Parameters
        -------
        tasks: list of strings
            list with the path to the ltl sketch for each task
        r_good: float
            success rate threshold to decide moving to the next task
        num_steps: int
            max number of steps per episode that the agent has to complete the task.
            if it does it, we consider a hit on its 'success rate'
            (this emulates considering the average reward after running a rollout for 'num_steps')
        min_steps: int
            min number of training steps per task required to the agent before moving on to another task
        total_steps: int
            total number of training steps that the agent has to learn all the tasks
        """
        self.tasks = tasks
        self.r_good = r_good
        self.num_steps = num_steps
        self.min_steps = min_steps
        self.total_steps = total_steps
        self.incremental = False

    def incremental_learning(self, incremental_steps):
        self.incremental = True
        self.total_steps += incremental_steps

    def restart(self):
        self.current_step = 0
        self.succ_rate: Mapping[str, MovingAverage] = {}
        for val in self.succ_rate.values():
            val.reset()
        self.current_task = -1

    def stop_learning(self):
        return self.total_steps <= self.current_step

    def get_current_step(self):
        return self.current_step

    def get_next_task(self):
        self.last_restart = -1
        self.current_task = (self.current_task+1) % len(self.tasks)
        return self.get_current_task()

    def get_current_task(self):
        return self.tasks[self.current_task]

    def add_step(self):
        self.current_step += 1

    def update_succ_rate(self, step, reward):
        t = self.get_current_task()
        if t not in self.succ_rate:
            self.succ_rate[t] = MovingAverage()
        if reward > 0 and (step-self.last_restart) <= self.num_steps:
            self.succ_rate[t].add(1.0)
        else:
            self.succ_rate[t].add(0.0)
        self.last_restart = step

    def stop_task(self, step):
        return False
        return self.min_steps <= step and self.r_good < self.get_succ_rate()

    def get_succ_rate(self):
        t = self.get_current_task()
        return self.succ_rate[t].get_moving_average()
