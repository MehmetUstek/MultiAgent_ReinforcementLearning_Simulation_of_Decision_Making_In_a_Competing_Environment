import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import gym
import gym.spaces as spaces

inputs = keras.Input(shape=(1,))
print(inputs.shape)
first_dense = layers.Dense(64, activation="relu")(inputs)
output_1 = layers.Dense(14)(first_dense)
# second_dense = layers.Dense(64, activation="relu")(first_dense)
# output_2 = layers.Dense(1)(second_dense)
model = keras.Model(inputs=inputs, outputs=output_1, name="_")
model.summary()
gym.envs.register(
    id='MyCombat-v0',
    entry_point='ma_gym.envs.combat:Combat',
    kwargs={'n_agents': 10,'n_food':20, 'full_observable': False, 'step_cost': -0.2}
    # It has a step cost of -0.2 now
)
env = gym.make('MyCombat-v0')
actions = env.action_space
#TODO: Turn this into np.array
# actions = list()
# for _ in range(env.n_agents):
#     actions.append(spaces.Discrete(14).n)

# def build_agent(model, action_shape):
#     policy = BoltzmannQPolicy()
#     memory = SequentialMemory(limit=50000, window_length=1)
#     dqn = DQNAgent(model=model, memory=memory, policy=policy,
#                    nb_actions=14, nb_steps_warmup=10, target_model_update=1e-2)
#     return dqn
# agent
# for agent in range(env.n_agents):
#     dqn = build_agent(model, actions)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)
# np.array(norm(env))
# print(env.observation_space)
# temp = list()
# for i in env.observation_space:
#     temp.append(i.shape)
# print(temp)
_obs_low = np.repeat(np.array([-1., 0., -1., 0., 0., 0.], dtype=np.float32), 10 * 10)
_obs_high = np.repeat(np.array([1., env.n_agents, env._init_health, 1., 1., 1.], dtype=np.float32),
                                   10 * 10)
print(_obs_low)
history = model.fit(_obs_low, _obs_high, epochs=10000)
# history = model.fit(env, batch_size=64, epochs=2, validation_split=0.2)
# scores = dqn.test(env, nb_episodes=100, visualize=False)
# print("Test loss:", test_scores[0])
# print("Test accuracy:", test_scores[1])