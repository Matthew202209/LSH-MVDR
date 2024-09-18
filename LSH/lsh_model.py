from typing import Optional
import torch.nn as nn
import torch
from .citadel_utils import PathManager


from transformers import AutoModelForMaskedLM, AutoConfig


class LSHEncoder(nn.Module):
    def __init__(
            self,
            model_path: str = "bert-base-uncased",
            dropout: float = 0.1,
            tok_projection_dim: Optional[int] = None,
            cls_projection_dim: Optional[int] = None,
    ):
        super().__init__()
        # remove recursive argument which is not supported now
        local_model_path = PathManager.get_local_path(model_path)
        cfg = AutoConfig.from_pretrained(local_model_path)
        cfg.output_hidden_states = True
        cfg.attention_probs_dropout_prob = dropout
        cfg.hidden_dropout_prob = dropout
        cfg.tok_projection_dim = tok_projection_dim
        self.transformer = AutoModelForMaskedLM.from_pretrained(local_model_path, config=cfg)

        self.cls_project = nn.Identity()  # to make torchscript happy
        if cls_projection_dim:
            linear = nn.Linear(cfg.hidden_size, cls_projection_dim)
            linear.weight.data.normal_(mean=0.0, std=0.02)
            self.cls_project = nn.Sequential(
                linear,
            )

        self.tok_project = nn.Identity()  # to make torchscript happy
        if tok_projection_dim:
            linear = nn.Linear(cfg.hidden_size, tok_projection_dim)
            linear.weight.data.normal_(mean=0.0, std=0.02)
            self.tok_project = nn.Sequential(
                linear,
            )

    def forward(self, tokens):
        ret = {}
        # make it transformer 4.x compatible
        outputs = self.transformer(**tokens, return_dict=True)
        hiddens = outputs.hidden_states[-1][:, 1:, :]
        attention_mask = tokens["attention_mask"][:, 1:]
        logits = outputs.logits[:, 1:, :]  # take out from the second one

        token_dense_repr = self.tok_project(hiddens) * attention_mask.unsqueeze(-1)
        token_dense_repr = token_dense_repr.clone().reshape(-1, 32)

        token_sparse_repr = torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)
        token_sparse_repr = token_sparse_repr.clone().reshape(-1, token_sparse_repr.shape[-1])
        cls_repr = self.cls_project(outputs.hidden_states[-1][:, 0, :])

        token_dense_mask = torch.any(token_dense_repr != 0, dim=1)
        token_sparse_mask = torch.any(token_sparse_repr != 0, dim=1)

        token_dense_repr = token_dense_repr[token_dense_mask]
        token_sparse_repr = token_sparse_repr[token_sparse_mask]

        # router_softmax_repr = torch.softmax(logits, dim=-1)
        ret["token_dense_repr"] = token_dense_repr.to(torch.float32)
        ret["token_sparse_repr"] = token_sparse_repr.to(torch.float32)
        ret["cls_repr"] = cls_repr.clone().to(torch.float32)
        ret["num_token"] =  torch.sum(attention_mask, dim=1)

        return ret