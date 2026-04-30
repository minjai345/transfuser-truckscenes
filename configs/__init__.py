"""Versioned TransFuser-TruckScenes configs.

각 `vN_*.py`는 base schema(`_base.py`)에서 override delta만 명시.
실험은 학습 스크립트의 `--config <name>` flag로 선택 (e.g. `--config v4_range`).
버전 내역은 `configs/README.md` 참조.
"""
from configs._base import TransfuserConfig

__all__ = ["TransfuserConfig", "load_config"]


def load_config(name: str) -> TransfuserConfig:
    """`configs/{name}.py`에서 `config` 변수를 로드.

    Args:
        name: 버전 파일 stem (e.g. "v3_baseline", "v4_range").
    Returns:
        해당 버전 파일의 `config` 인스턴스 (TransfuserConfig).
    """
    import importlib
    module = importlib.import_module(f"configs.{name}")
    return module.config
