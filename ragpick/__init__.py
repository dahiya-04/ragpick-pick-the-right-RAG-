"""ragpick: run a full RAG evaluation grid on your own documents with one command.

    ragpick run config.yaml

Python API:

    from ragpick import run
    run_dir = run("config.yaml")
"""

__version__ = "0.1.0"


def run(config_path, **kwargs):
    from .pipeline import run as _run

    return _run(config_path, **kwargs)
