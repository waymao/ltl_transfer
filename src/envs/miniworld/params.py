class GameParams:
    """
    Auxiliary class with the configuration parameters that the Game class needs
    """
    def __init__(self, 
                 map_fpath, 
                 prob, 
                 ltl_task, 
                 consider_night=False, 
                 init_dfa_state=0, 
                 init_loc=None,
        ):
        self.map_fpath = map_fpath
        self.prob = prob
        self.ltl_task = ltl_task
        self.consider_night = consider_night
        self.init_dfa_state = init_dfa_state
        self.init_loc = init_loc
