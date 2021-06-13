import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from gym.spaces import Tuple, Box
import gym
from gym import spaces

gym.envs.register(
    id='MyCombat-v0',
    entry_point='ma_gym.envs.combat:Combat',
    kwargs={'n_agents': 10,'n_food':20, 'full_observable': False, 'step_cost': -0.2}
    # It has a step cost of -0.2 now
)
env = gym.make('MyCombat-v0')
states_n = list()
for agent in range(env.n_agents):
    _obs_low = np.repeat(np.array([-1., 0., -1., 0., 0., 0.], dtype=np.float32), 10 * 10)
    _obs_high = np.repeat(np.array([1., env._n_opponents, env._init_health, 1., 1., 1.], dtype=np.float32),
                                       10 * 10)
    states = spaces.Box(_obs_low, _obs_high).shape
    # states = env.observation_space
    states_n.append(states)
# states = env.observation_space.shape
# states = env.observation_space.spaces[0].shape
# speeds = env.observation_space.spaces[1].shape
# shape = Box([states,speeds]).shape
actions = list()
# actions = spaces.Discrete(14).n
actions_n = list()
actions_temp = spaces.Discrete(14).n
for _ in range(env.n_agents):
    actions.append(spaces.Discrete(14).n)
# actions = env.action_space
# actions_shape = spaces.Discrete(14)

def build_model(states_n, actions):
    model = Sequential()
    # keras.Input(shape={1,10,600})
    model.add(Dense(3, activation='relu', input_shape=(1,10,600)))


    model.add(Dense(24, activation='relu'))
    model.add(Dense(24,activation='relu'))
    # model.add(Dense(24,activation='relu'))
    model.add(Flatten())
    for action in actions:
        model.add(Dense(action, activation='linear'))
    return model


model = build_model(states_n, actions)
model.summary()


def build_agent(model, actions_temp):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions_temp, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

# np.array([]) for action in actions
for action in actions:
    dqn = build_agent(model, action)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
scores = dqn.test(env, nb_episodes=100, visualize=False)
# print(np.mean(scores.history['episode_reward']))
_ = dqn.test(env, nb_episodes=15, visualize=True)