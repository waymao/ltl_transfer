from gymnasium import register
from .world import NavigateEnv
from .wrapper import MiniWorldLTLWrapper
from .non_visual_wrapper import NonVisualWrapper

register(
    id="LTLTransfer-Navigate-v0",
    entry_point="envs.miniworld.world:NavigateEnv",
)

__all__ = [
    'NavigateEnv', 'MiniWorldLTLWrapper', 'NonVisualWrapper'
]
