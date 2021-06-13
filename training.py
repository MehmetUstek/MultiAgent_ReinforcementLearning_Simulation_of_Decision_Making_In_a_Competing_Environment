import gym

gym.envs.register(
    id='MyCombat-v0',
    entry_point='ma_gym.envs.combat:Combat',
    kwargs={'n_agents': 10,'n_food':20, 'grid_shape':(25,25), 'full_observable': False, 'step_cost': -0.05}
    # It has a step cost of -0.2 now
)

NUM_EPISODES = 100
for episode in range(NUM_EPISODES):
    env = gym.make('MyCombat-v0')
    # monitor = monitor.Monitor(env=env, directory="D:/gym-results")
    # env.monitor.start("D:/tmp/gym-results",True)
    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0
    agentRewards = list()
    reward_n = list()
    obs_n = env.reset()
    while not all(done_n):
        env.render()
        obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
        # print(reward_n)
        ep_reward += sum(reward_n)
        temp_reward = reward_n
        for i in range(len(reward_n)):
            if len(agentRewards)<=i:
                agentRewards.insert(i, reward_n[i])
            else:
                agentRewards[i] += reward_n[i]

    print(agentRewards)
    print("sum reward", ep_reward)
    env.close()

