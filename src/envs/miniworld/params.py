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
                 init_angle=None,
                 step_rew=0.0,
                 succ_rew=1.0,
                 fail_rew=0.0,
                 max_edge_props=1,
        ):
        self.map_fpath = map_fpath
        self.prob = prob
        self.ltl_task = ltl_task
        self.consider_night = consider_night
        self.init_dfa_state = init_dfa_state
        self.init_loc = init_loc
        self.init_angle = init_angle

        self.step_rew = step_rew
        self.succ_rew = succ_rew
        self.fail_rew = fail_rew
        
        # edge-centric code
        self.max_edge_props = max_edge_props
