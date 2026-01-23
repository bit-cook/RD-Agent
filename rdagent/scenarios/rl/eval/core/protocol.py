"""


"""

from abc import abstractmethod
from rdagent.core.experiment import FBWorkspace


class BenchmarkBase:

    @abstractmethod
    def run(self, workspace: FBWorkspace) -> None:
        """
        It will run the evaluation process to evaluate the performance of the solution on the workspace.

        Output:
        - the evalouation result will saved into the workspace's filesystem
        """

    # Make a decision from following options:
    # - We encourage LLM to analyze results from the file system.
    # - We can provoide benchmark specific uitls to support LLM.
    #   - make a choice:
    #       - typed or not;
    #           - typed: Con: easily operate on the result
    #               class BenchmarkResult(BaseModel):
    #                   return_code: int
    #                   stdout: str
    #                   running_time: float
    #                   .....
    #            - untyped, e.g. string or json:
    #               Con: higly flexible.
    # def get_structured_result(self, workspace: FBWorkspace) -> ???:
