import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
import gym

gym.envs.register(
    id='MyCombat-v0',
    entry_point='ma_gym.envs.combat:Combat',
    kwargs={'n_agents': 10,'n_food':20, 'full_observable': False, 'step_cost': -0.2}
    # It has a step cost of -0.2 now
)
env = gym.make('MyCombat-v0')

def setup_and_train():
    # Create a single environment and register it
    def env_creator(_):
        return env

    single_env = env

    env_name = "Combat"
    register_env(env_name, env_creator)

    # Get environment obs, action spaces and number of agents
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    num_agents = single_env.n_agents

    # Create a policy mapping
    def gen_policy():
        return (None, obs_space, act_space, {})

    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return 'agent-' + str(agent_id)

    # Define configuration with hyperparam and training details
    config = {
        "log_level": "WARN",
        "monitor" : False,
        "num_workers": 3,
        "num_cpus_for_driver": 1,
        "num_cpus_per_worker": 1,
        "num_sgd_iter": 10,
        "train_batch_size": 10,
        "lr": 5e-3,
        # "gamma":0.99,
        "model": {"fcnet_hiddens": [8, 8]},
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "env": "SurvivalEnv"}

    # Define experiment details
    exp_name = 'my_exp'
    exp_dict = {
        'name': exp_name,
        'run_or_experiment': 'PPO',
        "stop": {
            "training_iteration": 10
        },
        'checkpoint_freq': 20,
        "config": config,
    }

    # Initialize ray and run
    ray.init()
    tune.run(**exp_dict)


if __name__ == '__main__':
    setup_and_train()
