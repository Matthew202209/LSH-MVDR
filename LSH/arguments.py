from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="./encode",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default="bert-base-uncased", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="./check_points",
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    token_dim: int = field(default=768)
    cls_dim: int = field(default=768)
    token_rep_relu: bool = field(default=False, )
    token_norm_after: bool = field(default=False)
    cls_norm_after: bool = field(default=False)
    x_device_negatives: bool = field(default=False)
    pooling: str = field(default='max')
    no_sep: bool = field(default=False, )
    no_cls: bool = field(default=False, )
    cls_only: bool = field(default=False, )
