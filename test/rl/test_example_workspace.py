import pytest
from pathlib import Path
from rdagent.utils.env import RLDockerEnv
from rdagent.scenarios.rl.eval.workspace import RLWorkspace

def test_example_workspace():
    # 1. Create an RLDockerEnv
    env = RLDockerEnv()
    env.prepare() # build the docker image

    # 2. Create an RLWorkspace
    workspace = RLWorkspace()

    # 3. Inject the code from rdagent/scenarios/rl/eval/example_workspace into the workspace
    example_workspace_path = Path(__file__).parent.parent.parent / "rdagent" / "scenarios" / "rl" / "eval" / "example_workspace"
    workspace.inject_code_from_folder(example_workspace_path)

    # 4. Run the workspace in the Docker environment
    result = workspace.run(env, "python main.py")

    # 5. Assert that the run was successful and the model file exists
    assert result.exit_code == 0
    model_file_path = workspace.workspace_path / "ppo_cartpole.zip"
    assert model_file_path.exists()
