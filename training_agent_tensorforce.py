from gym.envs import kwargs
from tensorforce.agents import Agent
import numpy as np
import gym
import gym.spaces as spaces

gym.envs.register(
    id='MyCombat-v0',
    entry_point='ma_gym.envs.combat:Combat',
    kwargs={'n_agents': 10,'n_food':20, 'full_observable': False, 'step_cost': -0.2}
    # It has a step cost of -0.2 now
)
env = gym.make('MyCombat-v0')
num_agents = env.n_agents
# obs_space = env.observation_space
action_space = env.action_space

states_n = list()
for agent in range(env.n_agents):
    _obs_low = np.repeat(np.array([-1., 0., -1., 0., 0., 0.], dtype=np.float32), 10 * 10)
    _obs_high = np.repeat(np.array([1., env._n_opponents, env._init_health, 1., 1., 1.], dtype=np.float32),
                                       10 * 10)
    states = spaces.Box(_obs_low, _obs_high).shape
    # states = env.observation_space
    states_n.append(states)
obs_space = states_n

# Define configuration for agent
config = dict(
    network=[
                dict(type='dense', size=8),
                dict(type='dense', size=8),
            ],
    optimizer=dict(type='adam', learning_rate=3e-4)
)

# Define Agents
agent_list = []
for i in range(num_agents):
    print("Creating Agent {}".format(i))
    agent1 = Agent.create(agent='tensorforce', states=obs_space, actions=action_space, parallel_interactions=5,environment=env, max_episode_timesteps=1000,
                          optimizer=dict(type='adam',learning_rate=3e-4), update=dict(unit='timesteps', batch_size=64), objective='policy_gradient', reward_estimation=dict(horizon=20))
    agent_list.append(agent1)


training_iterations = 10
batch_size = 5

# Create a batch of environments
env_batch = []
for i in range(batch_size):
    env_batch.append(env)

for i in range(training_iterations):

    # Inititalize some placeholders
    obs_batch = {i:[] for i in range(num_agents)}
    rew_batch = {i:[] for i in range(num_agents)}
    done_batch = {i:[] for i in range(num_agents)}
    action_batch = {b:{} for b in range(batch_size)}

    # Get initial obs for all envs
    for b in range(batch_size):
        obs = env_batch[b].reset()
        for agent_id in range(num_agents):
            obs_batch[agent_id].append(obs[agent_id])

    # Get actions for every agent on a batch of observations
    for agent_id in range(env.num_agents):
        current_agent = agent_list[agent_id]
        temp = obs_batch[agent_id]
        actions = current_agent.act(states=temp)
        for b in range(batch_size):
            action_batch[b][agent_id] = actions[b]

    # Step all environments based on action batch
    for b in range(batch_size):
        new_obs, rew, dones, info = env_batch[b].step(action_batch[b])
        for agent_id in env.state_n:
            rew_batch[agent_id].append(rew[agent_id])
            done_batch[agent_id].append(dones[agent_id])

    # Call observe to innternalize to experience trajectory
    for agent_id in new_obs:
        agent_list[agent_id].model.observe(reward = rew_batch[agent_id], terminal = done_batch[agent_id])

    # Log rewards
    if i%100==0:
        print("Reward for iteration {} is {}".format(i, np.mean(rew_batch[0])))