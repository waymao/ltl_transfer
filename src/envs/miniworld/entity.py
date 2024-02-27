from miniworld.entity import MeshEnt, COLOR_NAMES

class Key(MeshEnt):
    """
    Key the agent can pick up, carry, and use to open doors
    """

    def __init__(self, color, size):
        assert color in COLOR_NAMES
        self.color = color
        super().__init__(mesh_name=f"key_{color}", height=size, static=False)
