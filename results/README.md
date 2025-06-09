The `results/` directory stores simulation artifacts generated during tests and orchestrator runs.

* `*.log` files contain raw stdout/stderr produced by executed Python scripts.
* `*.summary` files record whether the run succeeded and are safe to share.

When used with the `Orchestrator`, summaries are copied here so the latest run outputs are easy to locate.
