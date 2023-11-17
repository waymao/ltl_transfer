from miniworld.entity import MeshEnt, COLOR_NAMES

class Key(MeshEnt):
    """
    Key the agent can pick up, carry, and use to open doors
    """

    def __init__(self, color):
        assert color in COLOR_NAMES
        self.color = color
        super().__init__(mesh_name=f"key_{color}", height=0.35, static=False)


class Ball(MeshEnt):
    """
    Ball (sphere) the agent can pick up and carry
    """

    def __init__(self, color, size=0.6):
        assert color in COLOR_NAMES
        self.color = color
        super().__init__(mesh_name=f"ball_{color}", height=size, static=False)
