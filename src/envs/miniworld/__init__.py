from gymnasium import register
from .world import NavigateEnv
from .world_no_vis import NavigateNoVisEnv, LidarNoVisEnv
from .wrapper import MiniWorldLTLWrapper, ProgressionTerminateWrapper
from .non_visual_wrapper import NonVisualWrapper, LidarWrapper

register(
    id="LTLTransfer-Navigate-v0",
    entry_point="envs.miniworld.world:NavigateEnv",
)

__all__ = [
    'NavigateEnv', 'MiniWorldLTLWrapper', 
    'NonVisualWrapper', 'ProgressionTerminateWrapper', 'NavigateNoVisEnv',
    'LidarWrapper', 'LidarNoVisEnv'
]
