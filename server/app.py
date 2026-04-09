import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openenv.core.env_server import create_fastapi_app
from env.environment import DataCleaningEnv, Action, Observation

app = create_fastapi_app(DataCleaningEnv, Action, Observation)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
