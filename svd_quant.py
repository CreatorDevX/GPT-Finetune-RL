import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List


def quantize_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    max_val = tensor.abs().max()
    if max_val == 0:
        return torch.zeros_like(tensor, dtype=torch.int8), torch.tensor(1.0, device=tensor.device)
    scale = max_val / 127.0
    q = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return q.to(dtype) * scale.to(dtype)


def svd_low_rank(weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    U, S, V = torch.linalg.svd(weight.float(), full_matrices=False)
    return U[:, :rank], S[:rank], V[:rank, :]


class SVDInt8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, U_q, U_scale, S_q, S_scale, V_q, V_scale, bias):
        dtype = x.dtype
        U = dequantize_int8(U_q, U_scale, dtype=dtype)
        S = dequantize_int8(S_q, S_scale, dtype=dtype)
        V = dequantize_int8(V_q, V_scale, dtype=dtype)

        ctx.save_for_backward(x, U, S, V)

        h = torch.matmul(x, V.t())
        h = h * S
        out = torch.matmul(h, U.t())

        if bias is not None:
            out = out + bias

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, U, S, V = ctx.saved_tensors

        if grad_output.dim() == 2:
            go = grad_output
        else:
            go = grad_output.reshape(-1, grad_output.size(-1))

        x_flat = x.reshape(-1, x.size(-1))

        dV = (S.unsqueeze(0) * (go @ U)).t() @ x_flat
        dU = torch.matmul(go.t(), x_flat @ V.t() * S).t()
        dS = (go @ U * (x_flat @ V.t())).sum(dim=0)

        dx = torch.matmul(go @ U * S, V)

        return dx, None, None, None, None, None, None, None


class SVDInt8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.register_buffer('U_q', torch.empty(out_features, rank, dtype=torch.int8))
        self.register_buffer('U_scale', torch.tensor(1.0))
        self.register_buffer('S_q', torch.empty(rank, dtype=torch.int8))
        self.register_buffer('S_scale', torch.tensor(1.0))
        self.register_buffer('V_q', torch.empty(rank, in_features, dtype=torch.int8))
        self.register_buffer('V_scale', torch.tensor(1.0))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SVDInt8LinearFn.apply(
            x, self.U_q, self.U_scale, self.S_q, self.S_scale,
            self.V_q, self.V_scale, self.bias,
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int) -> 'SVDInt8Linear':
        out_f, in_f = linear.weight.shape
        has_bias = linear.bias is not None
        module = cls(in_f, out_f, rank, bias=has_bias)
        module.set_from_weight(linear.weight.data)
        if has_bias:
            module.bias.copy_(linear.bias.data)
        return module

    def set_from_weight(self, weight: torch.Tensor):
        U_k, S_k, V_k = svd_low_rank(weight, self.rank)
        self.U_q, self.U_scale = quantize_int8(U_k)
        self.S_q, self.S_scale = quantize_int8(S_k)
        self.V_q, self.V_scale = quantize_int8(V_k)

    def get_reconstructed_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        U = dequantize_int8(self.U_q, self.U_scale, dtype=dtype)
        S = dequantize_int8(self.S_q, self.S_scale, dtype=dtype)
        V = dequantize_int8(self.V_q, self.V_scale, dtype=dtype)
        return torch.matmul(U * S, V)


def apply_svd_quant_to_model(model: nn.Module, rank: int = 128) -> nn.Module:
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, SVDInt8Linear.from_linear(child, rank))
        else:
            apply_svd_quant_to_model(child, rank)
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
            if isinstance(module, SVDInt8Linear):
                master_state[name] = module.get_reconstructed_weight(dtype=torch.float32).cpu()
        torch.save(master_state, self.master_path)

    def merge_adapter_and_save(self, model: nn.Module, output_path: str):
        master_state = torch.load(self.master_path, map_location='cpu')
        merged_state = {}

        for name, module in model.named_modules():
            if isinstance(module, SVDInt8Linear):
                master_w = master_state[name]
                current_w = module.get_reconstructed_weight(dtype=torch.float32).cpu()
                delta = current_w - master_w
                merged_state[name] = master_w + delta
                if module.bias is not None:
                    merged_state[f"{name}.bias"] = module.bias.clone().cpu()

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        torch.save(merged_state, output_path)

    def load_master_weights(self) -> Dict[str, torch.Tensor]:
        return torch.load(self.master_path, map_location='cpu')


def print_svd_info(model: nn.Module):
    total_params = 0
    svd_params = 0
    for module in model.modules():
        if isinstance(module, SVDInt8Linear):
            in_f = module.in_features
            out_f = module.out_features
            r = module.rank
            full = in_f * out_f
            low_rank = (in_f * r) + r + (r * out_f)
            total_params += full
            svd_params += low_rank
    reduction = (1 - svd_params / total_params) * 100 if total_params > 0 else 0
    print(f"SVD-Int8 base: {svd_params:,} params (full rank would be {total_params:,})")
    print(f"Backprop reduction: {reduction:.1f}%")
