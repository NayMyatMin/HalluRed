from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class LoraCfg:
    targets: List[str]
    r: int
    alpha: int
    dropout: float = 0.0


@dataclass
class WindowsCfg:
    early_len: int = 64
    use_answer_span: bool = True
    mode: str = "answer"  # one of: answer, early, full


@dataclass
class AdvCfg:
    epsilon: float = 1e-3


@dataclass
class LossCfg:
    lambda_: float = 0.3
    weights: dict = field(default_factory=dict)


@dataclass
class TelemetryCfg:
    nii_weighting: str = "magnitude"  # "none" or "magnitude"
    nii_min_norm: float = 1e-4        # ignore tokens with ||dX|| below this
    head_sample: int = 0              # 0 => mean over heads; >0 => first k heads


@dataclass
class TrainCfg:
    batch_size: int = 1
    max_steps: int = 1000
    eval_every: int = 200
    lr: float = 2e-4
    grad_accum_steps: int = 1
    save_dir: str = "checkpoints"
    save_every: int = 0  # 0 disables periodic save; always save at end


@dataclass
class DataCfg:
    corpus: str = "squad"  # or triviaqa, hotpotqa
    split: str = "train"
    limit: Optional[int] = 2000


@dataclass
class JoltCfg:
    model: str
    lora: LoraCfg
    windows: WindowsCfg
    adv: AdvCfg
    loss: LossCfg
    telemetry: TelemetryCfg
    train: TrainCfg
    data: DataCfg


def load_config(path: str) -> JoltCfg:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return JoltCfg(
        model=raw["model"],
        lora=LoraCfg(**raw["lora"]),
        windows=WindowsCfg(**raw["windows"]),
        adv=AdvCfg(**raw["adv"]),
        loss=LossCfg(**raw["loss"]),
        telemetry=TelemetryCfg(**raw.get("telemetry", {})),
        train=TrainCfg(**raw["train"]),
        data=DataCfg(**raw["data"]),
    )


