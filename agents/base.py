class BaseAgent:
    """Base agent skeleton.

    Parameters
    ----------
    name : str
        Identifier for the agent.
    tools : list[str] | None, optional
        Available tools for the agent, by default ``[]``.
    """

    def __init__(self, name: str, tools: list[str] | None = None) -> None:
        self.name = name
        self.tools = tools or []

    def act(self, context: dict) -> str:
        """Return an assistant message given the context.

        Subclasses should override this to implement behaviour.
        """
        raise NotImplementedError

