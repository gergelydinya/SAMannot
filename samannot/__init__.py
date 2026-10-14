"""SAMannot: Interactive video instance segmentation annotation tool based on SAM 2."""

try:
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_module

    if not GlobalHydra.instance().is_initialized():
        initialize_config_module("samannot.sam2_configs", version_base="1.2")
except Exception:
    pass


def main():
    """Launch the SAMannot GUI application."""
    from .main import main as _main
    return _main()


__version__ = "0.1.0"
__all__ = ["main"]
