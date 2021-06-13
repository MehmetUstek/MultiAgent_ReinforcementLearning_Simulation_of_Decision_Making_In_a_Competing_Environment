from gym.envs import kwargs
from tensorforce.agents import Agent, PPOAgent
import numpy as np
import gym
import gym.spaces as spaces
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from analyze import findlastepisode
trainhistdir = 'train_single_agent/'

test_model = False#False
continue_training = True#True
num_episodes = int(6e5)

gym.envs.register(
    id='MyCombat-v0',
    entry_point='ma_gym.envs.combat:Combat',
    kwargs={'n_agents': 10,'n_food':20, 'full_observable': False, 'step_cost': -0.2}
    # It has a step cost of -0.2 now
)
env = gym.make('MyCombat-v0')
num_agents = env.n_agents
# obs_space = env.observation_space
# action_space = env.action_space
#
# states_n = list()
# for agent in range(env.n_agents):
#     _obs_low = np.repeat(np.array([-1., 0., -1., 0., 0., 0.], dtype=np.float32), 10 * 10)
#     _obs_high = np.repeat(np.array([1., env._n_opponents, env._init_health, 1., 1., 1.], dtype=np.float32),
#                                        10 * 10)
#     states = spaces.Box(_obs_low, _obs_high).shape
#     # states = env.observation_space
#     states_n.append(states)
# obs_space = states_n

# Define configuration for agent
network_spec = dict(
    network=[
                dict(type='dense', size=8),
                dict(type='dense', size=8),
            ],
    optimizer=dict(type='adam', learning_rate=3e-4)
)

agent = PPOAgent(
    states=env.observation_space,
    actions=env.action_space,
    network=network_spec,
    batch_size=10,
    max_episode_timesteps=1000,
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=5000
    ),
    subsampling_fraction=0.1,
    # PPOAgent
    learning_rate=1e-3,
    multi_step=50,
    # Model

    discount=0.99,
    entropy_regularization=0.01,
    # PGModel
    baseline=None,

    # PGLRModel
    likelihood_ratio_clipping=0.2
)

# Create the runner
runner = Runner(agent=agent, environment=env)

# If testing or continue_training, load model parameters
if test_model or continue_training:
    runner.agent.restore_model(trainhistdir)

stat = np.zeros((num_episodes, 2))
global_episode_counter = 1

if continue_training:
    episode_offset = findlastepisode(trainhistdir)
    stat[:episode_offset, :] = np.load('{}Reward_stat_{}.npz'.format(trainhistdir, episode_offset))['stat'][
                               :episode_offset, :]


# Callback function printing episode statistics
def episode_finished(r):
    global global_episode_counter

    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))

    # store current statistics
    if not test_model:
        stat[r.episode - 1, :] = [r.episode_timestep, r.episode_rewards[-1]]

    if not test_model and (r.episode - 1) % SAVE_INT == 0:
        r.agent.save_model(trainhistdir)
        np.savez("{}Reward_stat_{}.npz".format(trainhistdir, r.episode), stat=stat)
        print("Model saved.")

    if global_episode_counter > num_episodes:
        return False
    else:
        global_episode_counter += 1
        return True


# Start learning
runner.run(episodes=num_episodes, max_episode_timesteps=200, episode_finished=episode_finished)
# runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)