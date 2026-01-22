try:
    from sglang._version import __version__, __version_tuple__
except ImportError:
    # Fallback for development without build
    __version__ = "v0.5.7"
    __version_tuple__ = (0, 5, 7)
