from abc import ABC, abstractmethod
from typing import List
from tsce_agent_demo.models.research_task import PaperMeta


class LiteratureSearchTool(ABC):
    @abstractmethod
    def run(self, query: str, k: int = 10) -> List[PaperMeta]:
        ...
