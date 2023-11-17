from gymnasium import register
from .world import MiniWorldEnv

register(
    id="LTLTransfer-Navigate-v0",
    entry_point="envs.miniworld.world:NavigateEnv",
)
