from .learning_params import LearningParameters
import torch

class Policy(torch.nn.Module):
    def __init__(self, 
            nn_module, 
            dfa,
            state_dim, 
            action_dim, 
            learning_params: LearningParameters,
            device="cpu"
        ):
        super().__init__()
        raise NotImplementedError()
    
    def forward(self, x):
        pass
    
    def update_target_network(self) -> None:
        """Synchronize the weight for the target network."""
        pass
    
    def compute_loss(
            self, 
            s1_NS, 
            a_N, 
            s2_NS, 
            r_N, 
            terminated_N, 
            next_q_index_N, 
            max_q_values_CN
        ):
        return 0

    def get_edge_labels(self):
        """
        Return proposition formula representing outgoing edges, e.g. a & b
        """
        return self.dfa.nodelist[self.dfa.ltl2state[self.ltl]].values()