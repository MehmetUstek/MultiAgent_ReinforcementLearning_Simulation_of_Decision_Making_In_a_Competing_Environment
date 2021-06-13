import logging

from gym import envs
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Register openai's environments as multi agent
# This should be done before registering new environments
env_specs = [env_spec for env_spec in envs.registry.all() if 'gym.envs' in env_spec.entry_point]
for spec in env_specs:
    register(
        id='ma_' + spec.id,
        entry_point='ma_gym.envs.openai:MultiAgentWrapper',
        kwargs={'name': spec.id, **spec._kwargs}
    )


register(
    id='Combat-v0',
    entry_point='ma_gym.envs.combat:Combat',
)

