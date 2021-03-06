"""Super Mario Bros for OpenAI Gym."""
import argparse
import gym
import copy
from nes_py.wrappers import JoypadSpace
from nes_py.app.play_human import play_human
from nes_py.app.play_random import play_random
from Q_Agent.play import play_q
from Q_Agent.play import play_double_q
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_AND_JUMP, TEST

# a key mapping of action spaces to wrap with
_ACTION_SPACES = {
    'right': RIGHT_ONLY,
    'simple': SIMPLE_MOVEMENT,
    'complex': COMPLEX_MOVEMENT,
    'right_and_jump': RIGHT_AND_JUMP,
    'test': TEST
}


def _get_args():
    """Parse command line arguments and return them."""
    parser = argparse.ArgumentParser(description=__doc__)
    # add the argument for the Super Mario Bros environment to run
    parser.add_argument('--env', '-e',
                        type=str,
                        default='SuperMarioBros-v0',
                        help='The name of the environment to play'
                        )
    # add the argument for the mode of execution as either human or random
    parser.add_argument('--mode', '-m',
                        type=str,
                        default='human',
                        choices=['human', 'random', 'q', 'deep_q', 'double_q'],
                        help='The execution mode for the emulation'
                        )
    # add the argument for adjusting the action space
    parser.add_argument('--actionspace', '-a',
                        type=str,
                        default='right_and_jump',
                        choices=['nes', 'right', 'right_and_jump', 'simple', 'complex', 'test'],
                        help='the action space wrapper to use'
                        )
    # add the argument for the number of steps to take in random mode
    parser.add_argument('--steps', '-s',
                        type=int,
                        default=500,
                        help='The number of random steps to take.',
                        )

    # parse arguments and return them
    return parser.parse_args()


def main():
    """The main entry point for the command line interface."""
    # parse arguments from the command line (argparse validates arguments)
    args = _get_args()
    # build the environment with the given ID
    env = gym.make(args.env)
    # wrap the environment with an action space if specified
    if args.actionspace != 'nes':
        # unwrap the actions list by key
        actions = _ACTION_SPACES[args.actionspace]
        # wrap the environment with the new action space
        env = JoypadSpace(env, actions)
    # play the environment with the given mode
    if args.mode == 'human':
        play_human(env)
    if args.mode == 'random':
        play_random(env, args.steps)
    if args.mode == 'q':
        play_q(env, args, actions)
    if args.mode == 'double_q':
        play_double_q(env, args, actions)

# explicitly define the outward facing API of this module
__all__ = [main.__name__]
