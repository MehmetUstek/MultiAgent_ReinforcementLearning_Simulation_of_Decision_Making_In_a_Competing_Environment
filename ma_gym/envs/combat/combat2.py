# -*- coding: utf-8 -*-

import copy
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, write_cell_text
from ...wrappers import monitor

logger = logging.getLogger(__name__)


class Combat(gym.Env):
    """
    We simulate a simple battle involving two opposing teams in a n x n grid.
    Each team consists of m = 5 agents and their initial positions are sampled uniformly in a 5 × 5
    square around the team center, which is picked uniformly in the grid. At each time step, an agent can
    perform one of the following actions: move one cell in one of four directions; attack another agent
    by specifying its ID j (there are m attack actions, each corresponding to one enemy agent); or do
    nothing. If agent A attacks agent B, then B’s health point will be reduced by 1, but only if B is inside
    the firing range of A (its surrounding 3 × 3 area). Agents need one time step of cooling down after
    an attack, during which they cannot attack. All agents start with 3 health points, and die when their
    health reaches 0. A team will win if all agents in the other team die. The simulation ends when one
    team wins, or neither of teams win within 40 time steps (a draw).

    The model controls one team during training, and the other team consist of bots that follow a hardcoded policy.
    The bot policy is to attack the nearest enemy agent if it is within its firing range. If not,
    it approaches the nearest visible enemy agent within visual range. An agent is visible to all bots if it
    is inside the visual range of any individual bot. This shared vision gives an advantage to the bot team.

    When input to a model, each agent is represented by a set of one-hot binary vectors {i, t, l, h, c}
    encoding its unique ID, team ID, location, health points and cooldown. A model controlling an agent
    also sees other agents in its visual range (3 × 3 surrounding area). The model gets reward of -1 if the
    team loses or draws at the end of the game. In addition, it also get reward of −0.1 times the total
    health points of the enemy team, which encourages it to attack enemy bots.

    Reference : Learning Multiagent Communication with Backpropagation
    Url : https://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation.pdf
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, grid_shape=(15, 15), n_agents=10, n_opponents=10, n_food=10, init_health=5,
                 full_observable=False,
                 step_cost=0.5, max_steps=1000, starving_limit=5):
        self.max_timesteps = max_steps
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._n_opponents = self.n_agents - 1
        self.init_food = n_food
        self.n_food = n_food
        self.number_of_observations = 5
        # food, agents etc.
        self.state_size = 5*5*self.n_agents * self.number_of_observations
        # self.state_size = grid_shape[0]*grid_shape[1]*self.n_agents * self.number_of_observations
        # self._n_opponents = n_opponents
        self._max_steps = max_steps
        self._step_cost = step_cost
        self._step_count = None
        self.food_id = 99999
        self.starving_limit_timestep = starving_limit
        self.starving_init = starving_limit
        self.agent_without_food = {_: None for _ in range(self.n_agents)}

        # self.monitor = monitor.Monitor(env=self, directory="D:/gym-results",force=True)
        self.action_space = 5 + self.n_agents
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.agent_prev_pos = {_: None for _ in range(self.n_agents)}

        self.food_pos = {_: None for _ in range(self.n_food)}
        # self.opp_pos = {_: None for _ in range(self.n_agents)}
        # self.opp_prev_pos = {_: None for _ in range(self.n_agents)}

        self._init_health = init_health
        self.agent_health = {_: None for _ in range(self.n_agents)}
        # self.opp_health = {_: None for _ in range(self._n_opponents)}
        self._agent_dones = [None for _ in range(self.n_agents)]
        self._agent_cool = {_: None for _ in range(self.n_agents)}
        # self._opp_cool = {_: None for _ in range(self._n_opponents)}
        self._total_episode_reward = None
        self.viewer = None
        self.full_observable = full_observable

        # 5 * 5 * (type, id, health, cool, x, y)
        # (food_id, id, health, cool, x, y)
        # id, cool, x, y
        # ten agents
        # 9 opponents
        # 10 foods
        # self._obs_low = np.repeat(np.array([-1., 0., -1., 0., 0., 0.], dtype=np.float32), self.n_agents * self.n_agents)
        # self._obs_high = np.repeat(np.array([1., self.n_agents, self._init_health, 1., 1., 1.], dtype=np.float32),
        #                            self.n_agents * self.n_agents)
        # self.observation_space = MultiAgentObservationSpace(
        #     [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])
        self.seed()

    def max_episode_timesteps(self):
        return self.max_timesteps

    def get_action_meanings(self, agent_i=None):
        action_meaning = []
        for _ in range(self.n_agents):
            meaning = [ACTION_MEANING[i] for i in range(self.n_agents)]
            meaning += ['Attack Opponent {}'.format(o) for o in range(self._n_opponents)]
            action_meaning.append(meaning)
        if agent_i is not None:
            assert isinstance(agent_i, int)
            assert agent_i <= self.n_agents
            return action_meaning[agent_i]
        else:
            return action_meaning

    @staticmethod
    def _one_hot_encoding(i, n):
        x = np.zeros(n)
        x[i] = 1
        return x.tolist()

    def isOpponent(self):
        pass

    def get_agent_obs(self):
        """
        When input to a model, each agent is represented by a set of one-hot binary vectors {i, t, l, h, c}
        encoding its unique ID, team ID, location, health points and cooldown.
        A model controlling an agent also sees other agents in its visual range (5 × 5 surrounding area).
        :return:
        """
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]

            # _agent_i_obs = self._one_hot_encoding(agent_i, self.n_agents)
            # _agent_i_obs += [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates
            # _agent_i_obs += [self.agent_health[agent_i]]
            # _agent_i_obs += [1 if self._agent_cool else 0]  # flag if agent is cooling down

            # team id , unique id, location,health, cooldown

            _agent_i_obs = np.zeros((self.number_of_observations, 5, 5))
            for row in range(0, 5):
                for col in range(0, 5):

                    if self.is_valid([row + (pos[0] - 2), col + (pos[1] - 2)]) and (
                            PRE_IDS['empty'] not in self._full_obs[row + (pos[0] - 2)][col + (pos[1] - 2)]):
                        x = self._full_obs[row + pos[0] - 2][col + pos[1] - 2]
                        # _type = 1 if PRE_IDS['agent'] in x else -1
                        if PRE_IDS['agent'] in x:
                            _type = 1
                        elif PRE_IDS['food'] in x:
                            _type = 0
                        else:
                            _type = -1
                        if not _type == 0:
                            _id = int(x[1:]) - 1  # id
                            # _agent_i_obs[0][row][col] = _type
                            _agent_i_obs[0][row][col] = self.agent_without_food[_id]
                            _agent_i_obs[1][row][col] = _id
                            _agent_i_obs[2][row][col] = self.agent_health[_id]
                            _agent_i_obs[3][row][col] = pos[0] / self._grid_shape[0]  # x-coordinate
                            _agent_i_obs[4][row][col] = pos[1] / self._grid_shape[1]  # y-coordinate

                        else:
                            _agent_i_obs[1][row][col] = self.food_id
                            _agent_i_obs[3][row][col] = pos[0] / self._grid_shape[0]  # x-coordinate
                            _agent_i_obs[4][row][col] = pos[1] / self._grid_shape[1]  # y-coordinate

            _agent_i_obs = _agent_i_obs.flatten().tolist()
            _obs.append(_agent_i_obs)

        return _obs

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_prev_pos[agent_i][0]][self.agent_prev_pos[agent_i][1]] = PRE_IDS['empty']
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)
        # self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['food']
        # print("full obs",agent_i,":",self._full_obs)

    def __update_agent_food_view(self, food_i):
        self._full_obs[self.food_pos[food_i][0]][self.food_pos[food_i][1]] = PRE_IDS['food']

    def __update_opp_view(self, opp_i):
        pass
        # self._full_obs[self.opp_prev_pos[opp_i][0]][self.opp_prev_pos[opp_i][1]] = PRE_IDS['empty']
        # self._full_obs[self.opp_pos[opp_i][0]][self.opp_pos[opp_i][1]] = PRE_IDS['opponent'] + str(opp_i + 1)

    def __init_full_obs(self):
        """ Each team consists of m = 5 agents and their initial positions are sampled uniformly in a 5 × 5
        square around the team center, which is picked uniformly in the grid.
        """
        self._full_obs = self.__create_grid()

        # select agent team center
        # Note : Leaving space from edges so as to have a 5x5 grid around it
        agent_team_center = self.np_random.randint(0, self._grid_shape[0]-1), self.np_random.randint(1,
                                                                                                       self._grid_shape[
                                                                                                           1] - 1)
        grid_center = self.np_random.randint(1, self._grid_shape[0] - 1), self.np_random.randint(1, self._grid_shape[
            1] - 1)



        # randomly select agent pos
        for agent_i in range(self.n_agents):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] ),
                       self.np_random.randint(0, self._grid_shape[1])]
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.agent_prev_pos[agent_i] = pos
                    self.agent_pos[agent_i] = pos
                    self.__update_agent_view(agent_i)
                    break
        # randomly select food pos
        for food_i in range(self.n_food):
            while True:
                food_pos = [self.np_random.randint(0, self._grid_shape[0]),
                            self.np_random.randint(0, self._grid_shape[1])]
                if self._full_obs[food_pos[0]][food_pos[1]] == PRE_IDS['empty']:
                    self.food_pos[food_i] = food_pos
                    self.__update_agent_food_view(food_i)
                    break

        self.__draw_base_img()

    def reset(self):
        self._step_count = 0
        self.agent_without_food = {_: 0 for _ in range(self.n_agents)}
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_health = {_: self._init_health for _ in range(self.n_agents)}
        # self.opp_health = {_: self._init_health for _ in range(self._n_opponents)}
        self._agent_cool = {_: True for _ in range(self.n_agents)}
        # self._opp_cool = {_: True for _ in range(self._n_opponents)}
        self._agent_dones = [False for _ in range(self.n_agents)]
        self.starving_limit_timestep = self.starving_init
        print("starv",self.starving_limit_timestep)
        self.n_food = self.init_food
        self.food_pos = {_: None for _ in range(self.n_food)}
        self.__init_full_obs()
        return self.get_agent_obs()

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        # draw agents
        for agent_i in range(self.n_agents):
            if self.agent_health[agent_i] > 0:
                fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=OPPONENT_COLOR)
                write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.3)

        # draw foods
        for food_i in range(self.n_food):
            if self.n_food>0 and self.food_pos[food_i]:
                fill_cell(img, self.food_pos[food_i], cell_size=CELL_SIZE, fill=FOOD_COLOR)
                write_cell_text(img, text=str(food_i + 1), pos=self.food_pos[food_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.3)

        img = np.asarray(img)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def __update_agent_pos(self, agent_i, move):
        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            # TODO: Do this
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None:
            if self._is_cell_vacant(next_pos):
                self.agent_pos[agent_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.__update_agent_view(agent_i)
                return False
            elif self._is_cell_food(next_pos):
                # print("yes")
                self.agent_pos[agent_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.__update_agent_view(agent_i)
                return True


    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def _is_cell_another_agent(self,pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['agent'])

    @staticmethod
    def is_visible(source_pos, target_pos):
        """
        Checks if the target_pos is in the visible range(5x5)  of the source pos

        :param source_pos: Coordinates of the source
        :param target_pos: Coordinates of the target
        :return:
        """
        return (source_pos[0] - 2) <= target_pos[0] <= (source_pos[0] + 2) \
               and (source_pos[1] - 2) <= target_pos[1] <= (source_pos[1] + 2)

    @staticmethod
    def is_fireable(source_pos, target_pos):
        """
        Checks if the target_pos is in the firing range(5x5)

        :param source_pos: Coordinates of the source
        :param target_pos: Coordinates of the target
        :return:
        """
        # print("source_pos",source_pos)
        # print("target_pos", target_pos)
        return (source_pos[0] - 1) <= target_pos[0] <= (source_pos[0] + 1) \
               and (source_pos[1] - 1) <= target_pos[1] <= (source_pos[1] + 1)


    def _is_cell_food(self, agent_pos1):
        # print("food_pos",self.food_pos)
        return self.is_valid(agent_pos1) and self._full_obs[agent_pos1[0]][agent_pos1[1]] == PRE_IDS['food']

    def step(self, agents_action):
        assert len(agents_action) == self.n_agents

        rewards = [0 for _ in range(self.n_agents)]

        self._step_count += 1
        if self._step_count % 5 == 0:
            self.starving_limit_timestep += 1
        # print("starving",self.starving_limit_timestep)
        for i in range(self.n_agents):
            if self.agent_health[i] > 0:
                rewards[i] = self._step_cost
        # rewards = [self._step_cost for _ in range(self.n_agents)]

        # # finding foods
        # for agent_i, action in enumerate(agents_action):
        #     if self._is_cell_food(self.agent_pos[agent_i]):
        #         print("yes", self.food_pos)
        #         # found food and deleted it.
        #         rewards[agent_i] += 5
        #         print("rewards after food", rewards)

        # processing attacks
        agent_health = copy.copy(self.agent_health)
        for agent_i, action in enumerate(agents_action):
            if agent_health[agent_i] > 0:
                if action > 4:  # attack actions
                    target_opp = action - 5
                    # if self.agent_health[target_opp] > 0:
                    if not agent_i == target_opp and agent_health[target_opp] > 0:
                        # TODO: Check if it is not itself.
                        if self.is_fireable(self.agent_pos[agent_i], self.agent_pos[target_opp]):
                            # opp_health[target_opp] -= 1
                            agent_health[target_opp] -= 1
                            rewards[agent_i] += SHOOTING_REWARD
                            rewards[target_opp] -= SHOOTING_REWARD
                            print(rewards)

                            if agent_health[target_opp] == 0:
                                # negative reward if it dies.
                                rewards[agent_i] += KILLING_REWARD
                                rewards[target_opp] += DEATH_PENALTY
                                pos = self.agent_pos[target_opp]
                                self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']
                            # print("health", agent_health)
                            # print(rewards)
                    self.agent_without_food[agent_i] += 1
                elif action <= 4:
                    retVal = self.__update_agent_pos(agent_i, action)
                    # rew = self.__update_agent_pos(agent_i, action)
                    # if rew:
                    #     rewards[agent_i] += rew
                    if retVal:
                        # print("yes", self.food_pos)
                        # found food and deleted it.
                        rewards[agent_i] += FOOD_REWARD
                        self.agent_without_food[agent_i] = 0
                        # self.n_food -= 1
                        # print(self.food_pos.items())
                        for key, value in self.food_pos.items():
                            val0 = value[0]
                            val1= value[1]
                            pos0= self.agent_pos[agent_i][0]
                            pos1 = self.agent_pos[agent_i][1]
                            if value[0] == self.agent_pos[agent_i][0] and value[1] == self.agent_pos[agent_i][1]:
                                # print("no")
                                self.food_pos[key] = (9999, 9999)
                        # print("rewards after food", rewards)
                    else:
                        self.agent_without_food[agent_i] += 1
                if agent_health[agent_i] > 0:
                    if self.agent_without_food[agent_i] >= self.starving_limit_timestep:
                        agent_health[agent_i] -= 1
                        if agent_health[agent_i] == 0:
                            rewards[agent_i] += DEATH_PENALTY

        self.agent_health = agent_health
        # print("health",self.agent_health)
        # process move actions
        # for agent_i, action in enumerate(agents_action):
        #     if self.agent_health[agent_i] > 0:
        #         if action <= 4:
        #             retVal = self.__update_agent_pos(agent_i, action)
        #             # rew = self.__update_agent_pos(agent_i, action)
        #             # if rew:
        #             #     rewards[agent_i] += rew
        #             if retVal:
        #                 # print("yes", self.food_pos)
        #                 # found food and deleted it.
        #                 rewards[agent_i] += FOOD_REWARD
        #                 self.agent_without_food[agent_i] = 0
        #                 # self.n_food -= 1
        #                 # print(self.food_pos.items())
        #                 for key, value in self.food_pos.items():
        #                     val0 = value[0]
        #                     val1= value[1]
        #                     pos0= self.agent_pos[agent_i][0]
        #                     pos1 = self.agent_pos[agent_i][1]
        #                     if value[0] == self.agent_pos[agent_i][0] and value[1] == self.agent_pos[agent_i][1]:
        #                         # print("no")
        #                         self.food_pos[key] = (9999, 9999)
        #                 # print("rewards after food", rewards)
        #             else:
        #                 self.agent_without_food[agent_i] += 1

        # Check whether agent is starving to death
        # for agent_i, action in enumerate(agents_action):
        #     if self.agent_health[agent_i] > 0:
        #         if self.agent_without_food[agent_i] >= self.starving_limit_timestep:
        #             self.agent_health[agent_i] -= 1
        #             if self.agent_health[agent_i] ==0:
        #                 rewards[agent_i] += DEATH_PENALTY


        # step overflow or all opponents dead
        temp_bool = [v == (9999, 9999) for k, v in self.food_pos.items()]
        # agent_health_check = [v for k, v in self.agent_health.items()]
        # a = 0
        # for i in agent_health_check:
        #     if i >= 1:
        #         a += 1
        #     if a > 1:
        #         agent_bool = False
        #         break
        # agent_bool = a <= 1
        if (self._step_count >= self._max_steps) or (sum([v for k, v in self.agent_health.items()]) == 0) \
                or all(temp_bool) :
            # TODO: closed (or agent_bool)

            self._agent_dones = [True for _ in range(self.n_agents)]
            for currentAgent in range(self.n_agents):
                if self.agent_health[currentAgent] > 0:
                    rewards[currentAgent] += SURVIVING_REWARD

        # print(rewards)
        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]
        # print("step",self._step_count)
        # print("health",self.agent_health)

        return self.get_agent_obs(), rewards, self._agent_dones, {'health': self.agent_health}

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


CELL_SIZE = 15

WALL_COLOR = 'black'
AGENT_COLOR = 'red'
OPPONENT_COLOR = 'blue'
FOOD_COLOR = 'yellow'

#Rewards
FOOD_REWARD = 3
SHOOTING_REWARD = 1
DEATH_PENALTY = -20
KILLING_REWARD = 2
SURVIVING_REWARD = 20

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'wall': 'W',
    'empty': '0',
    'agent': 'A',
    'opponent': 'X',
    'food': 'F',
}
