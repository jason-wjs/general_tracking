"""general_tracking package."""

__version__ = "0.1.0"

try:
  from . import tasks as _tasks  # noqa: F401
except Exception:
  _tasks = None

