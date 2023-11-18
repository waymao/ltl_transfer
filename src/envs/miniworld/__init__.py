from gymnasium import register
from .world import NavigateEnv
from .wrapper import MiniWorldLTLWrapper

register(
    id="LTLTransfer-Navigate-v0",
    entry_point="envs.miniworld.world:NavigateEnv",
)

__all__ = [
    'NavigateEnv', 'MiniWorldLTLWrapper'
]
