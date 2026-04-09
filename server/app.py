from openenv.core.env_server import create_fastapi_app
from env.environment import DataCleaningEnv, Action, Observation

app = create_fastapi_app(DataCleaningEnv, Action, Observation)
