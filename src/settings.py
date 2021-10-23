import os
from pathlib import Path

output: Path = Path(os.path.abspath(__file__)).parent.parent / "output"
