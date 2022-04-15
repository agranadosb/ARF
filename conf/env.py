from typing import Any
from dotenv import dotenv_values

vars: dict[str, Any] = {**dotenv_values()}
