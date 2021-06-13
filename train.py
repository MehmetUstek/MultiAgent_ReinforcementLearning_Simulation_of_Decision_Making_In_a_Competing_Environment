"""
Created on Wednesday Jan  16 2019
@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""

import numpy as np
import os
import random
import argparse
import pandas as pd
from training.dqn_agent import Agent
from training.brain import Brain
# from dqn_agent import Agent
import glob
import gym
N_AGENT = 5
MAX_TIMESTEP = 100
#TODO: Decrease replay steps
REPLAY_STEPS = 2
SAVE_STEPS = 30
#TODO: Increase food
N_FOOD = 50
STARVING_LIMIT = 6
STEP_COST = 0.5
GRID_SIZE = 10
ARG_LIST = ['learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_frequency', 'maximum_exploration',
            'max_timestep', 'first_step_memory', 'replay_steps', 'number_nodes', 'target_type', 'memory',
            'prioritization_scale', 'agents_number', 'grid_size']

gym.envs.register(
    id='MyCombat-v0',
    entry_point='ma_gym.envs.combat.combat2:Combat',
    kwargs={'n_agents': N_AGENT, 'grid_shape':(GRID_SIZE,GRID_SIZE), 'n_food':N_FOOD,'max_steps':MAX_TIMESTEP, 'full_observable': False, 'starving_limit':STARVING_LIMIT,'step_cost':STEP_COST}
    # It has a step cost of -0.2 now
)

def get_name_brain(args, idx):

    file_name_str = "Agent"+N_AGENT.__str__()+"Food"+N_FOOD.__str__()+"Replay"+REPLAY_STEPS.__str__()+"Starving"+STARVING_LIMIT.__str__()+"StepCost"+STEP_COST.__str__()+"Grid"+GRID_SIZE.__str__()

    return './results_agents/weights_files/' + file_name_str + '_' + str(idx) + '.h5'


def get_name_rewards(args):

    file_name_str = "Agent"+N_AGENT.__str__()+"Food"+N_FOOD.__str__()+"Replay"+REPLAY_STEPS.__str__()+"Starving"+STARVING_LIMIT.__str__()+"StepCost"+STEP_COST.__str__()+"Grid"+GRID_SIZE.__str__()

    return './results_agents/rewards_files/' + file_name_str + '.csv'


def get_name_timesteps(args):

    file_name_str = "Agent"+N_AGENT.__str__()+"Food"+N_FOOD.__str__()+"Replay"+REPLAY_STEPS.__str__()+"Starving"+STARVING_LIMIT.__str__()+"StepCost"+STEP_COST.__str__()+"Grid"+GRID_SIZE.__str__()

    return './results_agents/timesteps_files/' + file_name_str + '.csv'


class Environment(object):

    def __init__(self, arguments):
        current_path = os.path.dirname(__file__)  # Where your .py file is located
        self.env = gym.make('MyCombat-v0')
        self.episodes_number = arguments['episode_number']
        self.render = arguments['render']
        self.recorder = arguments['recorder']
        self.max_ts = arguments['max_timestep']
        self.test = arguments['test']
        self.filling_steps = arguments['first_step_memory']
        self.steps_b_updates = arguments['replay_steps']
        self.max_random_moves = arguments['max_random_moves']

        self.num_agents = arguments['agents_number']
        self.num_landmarks = self.num_agents
        self.grid_size = arguments['grid_size']

    def run(self, agents, file1, file2):

        total_step = 0
        rewards_list = []
        timesteps_list = []
        max_score = -10000
        max_agent_score = 0
        for episode_num in range(self.episodes_number):
            state = self.env.reset()
            if self.render:
                self.env.render()

            random_moves = random.randint(0, self.max_random_moves)

            # create randomness in initial state
            # for _ in range(random_moves):
            #     actions = [4 for _ in range(len(agents))]
            #     state, _, _ = self.env.step(actions)
            #     if self.render:
            #         self.env.render()

            # converting list of positions to an array
            # print(state)
            state = np.array(state)
            state = state.ravel()

            done = [False for _ in range(self.num_agents)]
            reward_all = 0
            time_step = 0
            print(done)
            while not all(done):

                # if self.render:
                #     self.env.render()
                actions = []
                for agent in agents:
                    actions.append(agent.greedy_actor(state))
                next_state, reward, done, info = self.env.step(actions)
                # converting list of positions to an array
                next_state = np.array(next_state)
                next_state = next_state.ravel()
                # print("next state",next_state)

                if not self.test:
                    for agent in agents:
                        agent.observe((state, actions, reward, next_state, done))
                        if total_step >= self.filling_steps:
                            agent.decay_epsilon()
                            # print("b_updates",self.steps_b_updates)
                            # print("time_step",time_step)
                            if time_step % self.steps_b_updates == 0:
                                # print("replay happened")
                                agent.replay()
                            agent.update_target_model()

                total_step += 1
                time_step += 1
                state = next_state
                reward_all += sum(reward)
                max_agent_score += max(reward)
                if self.render:
                    self.env.render()

            rewards_list.append(reward_all)
            timesteps_list.append(time_step)

            print("Episode {p}, Score: {s}, Final Step: {t}, Goal: {g}".format(p=episode_num, s=reward_all,
                                                                               t=time_step, g=done))

            if self.recorder:
                os.system("ffmpeg -r 2 -i ./results_agents/snaps/%04d.png -b:v 40000 -minrate 40000 -maxrate 4000k -bufsize 1835k -c:v mjpeg -qscale:v 0 "
                          + "./results_agents/videos/{a1}_{a2}.avi".format(a1=self.num_agents,a2=self.grid_size))
                files = glob.glob('./results_agents/snaps/*')
                for f in files:
                    os.remove(f)

            if not self.test:
                if episode_num % SAVE_STEPS == 0:
                    df = pd.DataFrame(rewards_list, columns=['score'])
                    df.to_csv(file1)

                    df = pd.DataFrame(timesteps_list, columns=['steps'])
                    df.to_csv(file2)

                    if total_step >= self.filling_steps:
                        if reward_all > max_score:
                            for agent in agents:
                                agent.brain.save_model()
                            max_score = reward_all


if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    # DQN Parameters
    parser.add_argument('-e', '--episode-number', default=1500, type=int, help='Number of episodes')
    #TODO: Change learning rate
    # parser.add_argument('-l', '--learning-rate', default=0.00005, type=float, help='Learning rate')
    parser.add_argument('-l', '--learning-rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='RMSProp',
                        help='Optimization method')
    parser.add_argument('-m', '--memory-capacity', default=1000000, type=int, help='Memory capacity')
    #TODO: Increase batch size make 128
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='Batch size')
    #TODO: Change target freq
    parser.add_argument('-t', '--target-frequency', default=1000, type=int,
                        help='Number of steps between the updates of target network')
    parser.add_argument('-x', '--maximum-exploration', default=100000, type=int, help='Maximum exploration step')
    parser.add_argument('-fsm', '--first-step-memory', default=0, type=float,
                        help='Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps', default=REPLAY_STEPS, type=float, help='Steps between updating the network')
    parser.add_argument('-nn', '--number-nodes', default=256, type=int, help='Number of nodes in each layer of NN')
    parser.add_argument('-tt', '--target-type', choices=['DQN', 'DDQN'], default='DDQN')
    parser.add_argument('-mt', '--memory', choices=['UER', 'PER'], default='UER')
    parser.add_argument('-pl', '--prioritization-scale', default=0.5, type=float, help='Scale for prioritization')

    parser.add_argument('-gn', '--gpu-num', default='2', type=str, help='Number of GPU to use')
    parser.add_argument('-test', '--test', action='store_false', help='Enable the test phase if "store_false"')

    # Game Parameters
    parser.add_argument('-k', '--agents-number', default=N_AGENT, type=int, help='The number of agents')
    parser.add_argument('-g', '--grid-size', default=GRID_SIZE, type=int, help='Grid size')
    parser.add_argument('-ts', '--max-timestep', default=MAX_TIMESTEP, type=int, help='Maximum number of timesteps per episode')

    # parser.add_argument('-rw', '--reward-mode', choices=[0, 1, 2], type=int, default=1, help='Mode of the reward,'
    #                                                                                          '0: Only terminal rewards'
    #                                                                                          '1: Partial rewards '
    #                                                                                          '(number of unoccupied landmarks'
    #                                                                                          '2: Full rewards '
    #                                                                                          '(sum of dinstances of agents to landmarks)')

    parser.add_argument('-rm', '--max-random-moves', default=0, type=int,
                        help='Maximum number of random initial moves for the agents')


    # Visualization Parameters
    parser.add_argument('-r', '--render', action='store_false', help='Turn on visualization if "store_false"')
    # parser.add_argument('-r', '--render', action='store_false', help='Turn on visualization if "store_false"')
    parser.add_argument('-re', '--recorder', action='store_true', help='Store the visualization as a movie '
                                                                       'if "store_false"')

    args = vars(parser.parse_args())
    # os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_num']
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    env = Environment(args)

    state_size = env.env.state_size
    action_space = env.env.action_space

    all_agents = []
    # TODO: Remember to change this.
    # brain_file = get_name_brain(args, 0)
    # brain = Brain(state_size, action_space, brain_file, args)
    for b_idx in range(args['agents_number']):
        brain_file = get_name_brain(args, b_idx)
        all_agents.append(Agent(state_size, action_space, b_idx, brain_file, args))
        # all_agents.append(Agent(state_size, action_space, b_idx, brain_file, args,brain))

    rewards_file = get_name_rewards(args)
    timesteps_file = get_name_timesteps(args)

    env.run(all_agents, rewards_file, timesteps_file)