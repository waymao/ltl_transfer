#!/usr/bin/env python3

"""
from miniworld / manual_control.py
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import argparse

import gymnasium as gym
from envs.miniworld.wrapper import MiniWorldLTLWrapper
from envs.miniworld.params import GameParams

import miniworld
from miniworld.manual_control import ManualControl

class CustomManualControl(ManualControl):
    def step(self, action):
        result = super().step(action)
        print("True Propositions:", self.env.get_true_propositions())
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="MiniWorld-Hallway-v0")
    parser.add_argument(
        "--domain-rand", action="store_true", help="enable domain randomization"
    )
    parser.add_argument(
        "--no-time-limit", action="store_true", help="ignore time step limits"
    )
    # parser.add_argument(
    #     "--top_view",
    #     action="store_true",
    #     default=True
    #     help="show the top view instead of the agent view",
    # )
    args = parser.parse_args()
    # view_mode = "top" if args.top_view else "agent"
    view_mode = "top"

    env = gym.make(args.env_name, view=view_mode, render_mode="human")

    ltl_formula = ('until', 'True', "a")
    params = GameParams('../../../experiments/maps/map_0.txt', 1, ltl_formula, False, False, None)
    wrapped = MiniWorldLTLWrapper(env, params)
    miniworld_version = miniworld.__version__

    print(f"Miniworld v{miniworld_version}, Env: {args.env_name}")

    manual_control = CustomManualControl(wrapped, args.no_time_limit, args.domain_rand)
    manual_control.run()


if __name__ == "__main__":
    main()
