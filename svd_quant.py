import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


def svd_low_rank(weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    U, S, V = torch.linalg.svd(weight.float(), full_matrices=False)
    return U[:, :rank], S[:rank], V[:rank, :]


class SVDLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, U, S, V, bias):
        dtype = x.dtype
        W = torch.matmul(U.to(dtype) * S.to(dtype), V.to(dtype))
        ctx.save_for_backward(x, U, S, V)
        return F.linear(x, W, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, U, S, V = ctx.saved_tensors
        dtype = x.dtype
        U, S, V = U.to(dtype), S.to(dtype), V.to(dtype)

        if grad_output.dim() == 2:
            go = grad_output
        else:
            go = grad_output.reshape(-1, grad_output.size(-1))

        x_flat = x.reshape(-1, x.size(-1))

        dU = torch.matmul(go.t(), x_flat @ V.t() * S).t()
        dS = (go @ U * (x_flat @ V.t())).sum(dim=0)
        dV = (S.unsqueeze(0) * (go @ U)).t() @ x_flat

        dx = torch.matmul(go @ U * S, V)

        dbias = go.sum(dim=0) if ctx.needs_input_grad[4] else None

        return dx, dU, dS, dV, dbias


class SVDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.register_buffer('U', torch.empty(out_features, rank))
        self.register_buffer('S', torch.empty(rank))
        self.register_buffer('V', torch.empty(rank, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SVDLinearFn.apply(x, self.U, self.S, self.V, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int) -> 'SVDLinear':
        out_f, in_f = linear.weight.shape
        has_bias = linear.bias is not None
        module = cls(in_f, out_f, rank, bias=has_bias)
        module.set_from_weight(linear.weight.data)
        if has_bias:
            module.bias.data.copy_(linear.bias.data)
        return module

    def set_from_weight(self, weight: torch.Tensor):
        U_k, S_k, V_k = svd_low_rank(weight, self.rank)
        self.U.copy_(U_k)
        self.S.copy_(S_k)
        self.V.copy_(V_k)

    def get_reconstructed_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.matmul(self.U.to(dtype) * self.S.to(dtype), self.V.to(dtype))


def _replace_linears(module: nn.Module, rank: int, progress_bar):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, SVDLinear.from_linear(child, rank))
            progress_bar.update(1)
        else:
            _replace_linears(child, rank, progress_bar)


def apply_svd_to_model(model: nn.Module, rank: int = 128) -> nn.Module:
    from tqdm.auto import tqdm
    total_linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    pbar = tqdm(total=total_linears, desc="SVD factorizing")
    _replace_linears(model, rank, pbar)
    pbar.close()
    return model


class MasterWeightManager:
    def __init__(self, model: nn.Module, master_path: str):
        self.master_path = master_path
        master_dir = os.path.dirname(master_path)
        if master_dir:
            os.makedirs(master_dir, exist_ok=True)
        self._save_master_weights(model)

    def _save_master_weights(self, model: nn.Module):
        master_state = {}
        for name, module in model.named_modules():
            if isinstance(module, SVDLinear):
                master_state[name] = module.get_reconstructed_weight(dtype=torch.float32).cpu()
        torch.save(master_state, self.master_path)

    def merge_and_save(self, model: nn.Module, output_path: str):
        master_state = torch.load(self.master_path, map_location='cpu')
        merged_state = {}

        for name, module in model.named_modules():
            if isinstance(module, SVDLinear):
                master_w = master_state[name]
                current_w = module.get_reconstructed_weight(dtype=torch.float32).cpu()
                delta = current_w - master_w
                merged_state[name] = master_w + delta
                if module.bias is not None:
                    merged_state[f"{name}.bias"] = module.bias.clone().cpu()

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(merged_state, output_path)

    def load_master_weights(self) -> Dict[str, torch.Tensor]:
        return torch.load(self.master_path, map_location='cpu')


def print_svd_info(model: nn.Module):
    total_params = 0
    svd_params = 0
    for module in model.modules():
        if isinstance(module, SVDLinear):
            in_f = module.in_features
            out_f = module.out_features
            r = module.rank
            full = in_f * out_f
            low_rank = (in_f * r) + r + (r * out_f)
            total_params += full
            svd_params += low_rank
    reduction = (1 - svd_params / total_params) * 100 if total_params > 0 else 0
    print(f"SVD base: {svd_params:,} params (full rank would be {total_params:,})")
    print(f"Backprop reduction: {reduction:.1f}%")
