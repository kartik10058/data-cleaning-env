from openenv.core.env_server import create_fastapi_app
from env.environment import DataCleaningEnv
from env.environment import Action, Observation

env = DataCleaningEnv(task_name="easy")
app = create_fastapi_app(env, Action, Observation)
