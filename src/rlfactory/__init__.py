"""RLFactory: verifier-driven infrastructure for LLM post-training.

The package is organised as a clean architecture:

* ``rlfactory.domain`` holds framework-free records (tasks, trajectories, rewards, runs).
* Application services (``tasks``, ``rollouts``, ``datasets``, ``evaluation``, ``experiments``,
  ``flywheel``) orchestrate the domain through protocols.
* Adapters (``sandbox``, ``policies``, ``trajectories``, ``training``, ``remote``) bind those
  protocols to concrete technology (subprocess/Docker, HTTP model servers, Parquet/DuckDB, TRL, SSH).
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
