from dataclasses import dataclass
from typing import List
import os
import time

@dataclass
class SuccEntry:
    """
    An entry of transfer / relabel result
    """
    # some default values are placeholder to be filled at the end
    # of an epi.
    ltl_task: list
    init_x: float
    init_y: float
    success: bool = False
    final_ltl: list = ""
    epi_len: int = -1
    global_step: int = -1
    time: float = -1
    ltl_deadend: bool = False
    saved: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = time.time()

    def to_list(self) -> list:
        return [
            "\"" + str(self.ltl_task) + "\"", 
            self.init_x, self.init_y, 
            self.success, 
            "\"" + str(self.final_ltl) + "\"", 
            self.ltl_deadend,
            self.epi_len, 
            self.global_step, self.time]



class SuccLogger:
    def __init__(self, log_path):
        self.log_file = log_path + os.sep + "success_map.csv"
        self.db: List[SuccEntry] = []
        with open(self.log_file, 'w') as f:
            f.write("task,x,y,success,final_ltl,ltl_deadend,len,global_step,time\n")

    def new_epi(self):
        pass

    def report_result(self, entry: SuccEntry):
        self.db.append(entry)
    
    def flush(self):
        self.db = []

    def save(self):
        with open(self.log_file, 'a') as f:
            for entry in self.db:
                if not entry.saved:
                    str_list = [str(e) for e in entry.to_list()]
                    f.write(",".join(str_list) + '\n')
                    entry.saved = True
