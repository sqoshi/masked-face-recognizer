from typing import Generator, List

from config.run_configuration import Configuration


class GroupConfiguration:
    """Groups multiple configurations into a group."""

    def __init__(
        self, name: str, configs: List[Configuration], description: str = ""
    ) -> None:
        self.name = name
        self.description = description
        self.config_list = configs

    def __iter__(self) -> Generator[Configuration, None, None]:
        """Iterator yields configurations stored in group."""
        for element in self.config_list:
            yield element
