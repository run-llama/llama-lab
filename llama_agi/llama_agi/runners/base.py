from abc import abstractmethod
from typing import List, Optional

from llama_agi.execution_agent.base import BaseExecutionAgent
from llama_agi.task_manager.base import BaseTaskManager


class BaseAGIRunner:
    def __init__(
        self, task_manager: BaseTaskManager, execution_agent: BaseExecutionAgent
    ) -> None:
        self.task_manager = task_manager
        self.execution_agent = execution_agent

    @abstractmethod
    def run(
        self,
        objective: str,
        initial_task: str,
        sleep_time: int,
        initial_task_list: Optional[List[str]] = None,
    ) -> None:
        """Run the task manager and execution agent in a loop."""
