# https://github.com/MohammadrezaBanaei/LoRA-XS/blob/main/utils/initialization_utils.py
from typing import Tuple

from contextlib import contextmanager
import math
import re
import types
import functools

# import bitsandbytes as bnb
import peft
import torch
from peft.import_utils import is_bnb_available
from peft.utils import _get_submodules
from torch.nn import init
from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD
import numpy as np
from typing import Tuple

import torch.nn.functional as F

class RandomLinear(torch.nn.Module):
    def __init__(self, r, r_trainable):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(r_trainable))
        self.random_weight = torch.nn.Parameter(torch.randn(r_trainable, r, r), requires_grad=False)
        self.r_trainable = r_trainable
    
    @property
    def effective_weight(self) -> torch.Tensor:
        return torch.einsum("b,bij->ij", self.weight, self.random_weight) / math.sqrt(self.r_trainable)
    
    def forward(self, x):
        return x @ self.effective_weight

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


@contextmanager
def disable_adapter(self):
    n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    adapter_modules = []
    for module in self.modules():
        if hasattr(module, "_disable_adapters"):
            adapter_modules.append((module, module._disable_adapters))
    try:
        # Turn off adapter layers for forward/backward while inside the CM
        self.base_model.disable_adapter_layers()
        yield
    finally:
        self.base_model.enable_adapter_layers()
        for active_adapter, is_disabled in adapter_modules:
            active_adapter.enable_adapters(not is_disabled)
            if hasattr(active_adapter, "lora_A"):
                active_adapter.lora_A.default.weight.requires_grad = False  # only the r*r matrix will be tuned
            if hasattr(active_adapter, "lora_B"):
                active_adapter.lora_B.default.weight.requires_grad = False  # only the r*r matrix will be tuned
    new_n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

def lora_get_delta_weight_faster_chatgpt(self, adapter) -> torch.Tensor:
    # Shapes: A:(r,n), B:(m,r), M:(r,r), out:(m,n)
    B_raw = self.lora_B[adapter].weight
    A_raw = self.lora_A[adapter].weight

    device = B_raw.device
    dtype  = B_raw.dtype
    cast_to_fp32 = (device.type == "cpu" and dtype == torch.float16)
    work_dtype = torch.float32 if cast_to_fp32 else dtype

    # --- per-adapter cache bucket ---
    ckey = f"_merge_cache_{adapter}"
    cache = getattr(self, ckey, None)
    if cache is None:
        cache = {}
        setattr(self, ckey, cache)

    # --- cache frozen A/B as contiguous tensors in final compute dtype/device ---
    A = cache.get("A")
    B = cache.get("B")
    if A is None or A.device != device or A.dtype != work_dtype or not A.is_contiguous():
        A = A_raw.to(device=device, dtype=work_dtype, copy=False).contiguous()
        cache["A"] = A
        cache["AT"] = A.transpose(0, 1).contiguous()  # (n, r), useful when we pick (B@M)@A
    if B is None or B.device != device or B.dtype != work_dtype or not B.is_contiguous():
        B = B_raw.to(device=device, dtype=work_dtype, copy=False).contiguous()
        cache["B"] = B

    A = cache["A"]
    AT = cache["AT"]
    B = cache["B"]

    r, n = A.shape
    m, rB = B.shape
    if r != rB:
        raise ValueError(f"Rank mismatch: A is (r={r}, n={n}) but B is (m={m}, r={rB})")

    # --- scratch buffers reused every call ---
    buf_mn = cache.get("buf_mn")
    if buf_mn is None or buf_mn.shape != (m, n) or buf_mn.dtype != work_dtype or buf_mn.device != device:
        buf_mn = torch.empty((m, n), device=device, dtype=work_dtype)
        cache["buf_mn"] = buf_mn

    buf_rn = cache.get("buf_rn")
    if buf_rn is None or buf_rn.shape != (r, n) or buf_rn.dtype != work_dtype or buf_rn.device != device:
        buf_rn = torch.empty((r, n), device=device, dtype=work_dtype)
        cache["buf_rn"] = buf_rn

    buf_mr = cache.get("buf_mr")
    if buf_mr is None or buf_mr.shape != (m, r) or buf_mr.dtype != work_dtype or buf_mr.device != device:
        buf_mr = torch.empty((m, r), device=device, dtype=work_dtype)
        cache["buf_mr"] = buf_mr

    # --- mapping tensor in place (no module casting) ---
    # Handle RandomLinear case
    if hasattr(self.default_lora_latent_mapping, 'random_weight'):
        # Compute the effective diagonal weight for RandomLinear
        M = self.default_lora_latent_mapping.effective_weight.to(device=device, dtype=work_dtype, copy=False).contiguous()
    else:
        M = self.default_lora_latent_mapping.weight.to(device=device, dtype=work_dtype, copy=False).contiguous()
    
    if M.dim() != 2 or M.shape != (r, r):
        raise ValueError(f"default_lora_latent_mapping must be square (r×r); got {tuple(M.shape)}")

    scale = self.scaling[adapter]

    with torch.no_grad():
        # Heuristic: choose association that touches the smaller of m vs n first.
        # Both do 2 GEMMs; this lightly helps cache locality.
        if m <= n:
            # (B @ M) @ A
            torch.matmul(B, M, out=buf_mr)         # (m, r)
            torch.matmul(buf_mr, A, out=buf_mn)    # (m, n)
        else:
            # B @ (M @ A)
            torch.matmul(M, A, out=buf_rn)         # (r, n)
            torch.matmul(B, buf_rn, out=buf_mn)    # (m, n)

        res = transpose(buf_mn, self.fan_in_fan_out).mul_(scale)
        if cast_to_fp32:
            res = res.to(dtype=dtype)

        # return a copy so future calls don't mutate our reused buffers
        return res.clone()


def lora_get_delta_weight(self, adapter) -> torch.Tensor:
    # This function is introduced in newer PEFT versions. we modify this function instead of modifying
    # the merge function (as we did previously for version 0.4.0 of PEFT).
    """
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_B[adapter].weight.dtype

    # In case users wants to merge the adapter weights that are in
    # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()
    
    default_lora_latent_mapping = self.default_lora_latent_mapping.to(weight_A.dtype)
    
    # Handle RandomLinear case
    if hasattr(default_lora_latent_mapping, 'random_weight'):
        # Compute the effective diagonal weight for RandomLinear
        effective_weight = default_lora_latent_mapping.effective_weight
        output_tensor = transpose(
            weight_B @ effective_weight @ weight_A,
            self.fan_in_fan_out
        ) * self.scaling[adapter]
    elif default_lora_latent_mapping.weight.dim() == 2:
        output_tensor = transpose(
            weight_B @ default_lora_latent_mapping.weight @ weight_A,
            self.fan_in_fan_out
        ) * self.scaling[adapter]
    elif default_lora_latent_mapping.weight.dim() == 1:
        output_tensor = transpose(
            (weight_B * default_lora_latent_mapping.weight) @ weight_A,
            self.fan_in_fan_out
        ) * self.scaling[adapter]
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {default_lora_latent_mapping.weight.dim()}D")

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

        # cast back the weights
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)

    return output_tensor


def lora_forward_latent(self, x: torch.Tensor):
    previous_dtype = x.dtype

    if self.active_adapter[0] not in self.lora_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter[0]] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    elif self.r[self.active_adapter[0]] > 0 and not self.merged:
        result = F.linear(x, transpose(self.weight.to(x.dtype), self.fan_in_fan_out), bias=self.bias)

        x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

        # adding latent_mapping in the forward loop
        x = self.lora_dropout[self.active_adapter[0]](x)
        x = self.lora_A[self.active_adapter[0]](x)
        
        # Handle both 1D and 2D latent mapping
        latent_weight = self.default_lora_latent_mapping.weight.to(x.dtype).contiguous()
        # with FSDP.summon_full_params(self):
        if latent_weight.dim() == 2:
            x = F.linear(x, latent_weight)
        elif latent_weight.dim() == 1:
            x = self.default_lora_latent_mapping(x)
            # x = F.linear(x, self.default_lora_latent_mapping.effective_weight)
        else:
            raise ValueError(f"Expected 1D or 2D latent mapping, got {latent_weight.dim()}D")
        
        result += (
            self.lora_B[self.active_adapter[0]](x)
            * self.scaling[self.active_adapter[0]]
        )
    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    result = result.to(previous_dtype)

    return result

def run_svd_sklearn(input_matrix: np.ndarray, rank: int, n_iter: int, random_state: int) -> Tuple[np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=rank, n_iter=n_iter, random_state=random_state)
    svd.fit(input_matrix)
    reduced_matrix = svd.transform(input_matrix)
    return reduced_matrix, svd


def get_linear_rec_svd_sklearn(input_matrix: np.ndarray, rank: int, n_iter: int,
                       random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced_matrix, svd = run_svd_sklearn(input_matrix, rank, n_iter, random_state)

    reconstructed_matrix = svd.inverse_transform(reduced_matrix)
    return reconstructed_matrix, reduced_matrix, svd.components_

def get_linear_rec_svd(input_matrix: torch.Tensor, rank: int, n_iter: int,
                       random_state: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    
    # Perform truncated SVD
    U, S, V = torch.svd_lowrank(input_matrix, q=rank, niter=n_iter)
    
    # Transform: project to reduced space
    reduced_matrix = U @ torch.diag(S)
    
    # Inverse transform: reconstruct from reduced space
    reconstructed_matrix = reduced_matrix @ V.T
    
    # Components (right singular vectors, transposed)
    components = V.T
    
    return reconstructed_matrix, reduced_matrix, components


def get_replacement_module(weight, module_name, type, reconstruct_config):
    cfg = reconstruct_config[type]
    if type == 'svd':
        reconstructed_matrix, enc, dec = get_linear_rec_svd(
            # input_matrix=weight.float().cpu().detach().numpy(), 
            input_matrix=weight.float(), 
            rank=cfg['rank'], 
            n_iter=cfg['n_iter'], 
            random_state=cfg['random_state']
        )
        final_enc = torch.tensor(enc, dtype=weight.dtype, device=weight.device)
        final_dec = torch.tensor(dec, dtype=weight.dtype, device=weight.device)
    else:
        raise NotImplementedError(f"{type} is currently not supported.")
    return final_enc, final_dec


def init_module_weights(target_module: torch.nn.Linear, sigma: float):
    # Initialize weights with Gaussian distribution
    torch.nn.init.normal_(target_module.weight, mean=0, std=sigma)
    if hasattr(target_module, "bias"):
        # Set bias to zeros
        if target_module.bias is not None:
            torch.nn.init.zeros_(target_module.bias)


def replace_module_weights(target_module, new_weight):
    device = target_module.weight.device
    target_module.weight = torch.nn.Parameter(new_weight.to(target_module.weight.dtype)).to(device)

    # dispatch to correct device
    # for name, module in target_module.named_modules():
    #     if "lora_" in name:
    #         module.to(device)


def update_decoder_weights(target_module, new_weight):
    device = target_module.weight.device
    with torch.no_grad():
        target_module.weight.copy_(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def kaiming_uniform_init_lower_half(matrix: torch.tensor):
    rows, _ = matrix.size()
    init.kaiming_uniform_(matrix[math.ceil(rows / 2):, :], a=math.sqrt(5))
    return matrix

def kaiming_uniform_init(matrix: torch.tensor):
    init.kaiming_uniform_(matrix, a=math.sqrt(5))
    return matrix
  
def find_and_initialize_lora_xs(model, lora_config, adapter_name, reconstr_type, reconstruct_config):
    """
    :param adapter_name: options: 'default'
    :param reconstr_type: options: 'svd'
    """
    learn_diagonal_only = reconstruct_config['learn_diagonal_only']
    half_init_dec = reconstruct_config['half_init_dec']
    replacement_module_random_init = reconstruct_config['replacement_module_random_init']
    reconstruction_mode = reconstruct_config['reconstr_mode']
    r_squared = reconstruct_config['r_squared']  # whether using r*r matrix between lora_A and lora_B or not
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    if loaded_in_8bit and not is_bnb_available():
        raise ImportError(
            "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
            "You can install it with `pip install bitsandbytes`."
        )
    key_list = [key for key, _ in model.named_modules()]
    assert (not isinstance(lora_config.target_modules, str))
    print("Iterating through model's specified modules to initialize A/B matrices. [target_modules]", lora_config.target_modules)
    all_linear_names = []
    all_linears = []
    pbar = tqdm(key_list, desc='Initializing Lora-XS')
    num_lora_xs_modules = 0
    model.disable_adapter = types.MethodType(disable_adapter, model)

    for key in pbar:
        target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
        if target_module_found:
            num_lora_xs_modules += 1
            _, target, target_name = _get_submodules(model, key)
            linear_module_cls = {
                "fp32": functools.partial(torch.nn.Linear, dtype=torch.float32),
                "fp16": functools.partial(torch.nn.Linear, dtype=torch.float16),
                "bf16": functools.partial(torch.nn.Linear, dtype=torch.bfloat16),
                # "fp8": bnb.nn.Linear8bitLt,
                # "fp4": bnb.nn.Linear4bit,
            }
            LINEAR_MODULE = linear_module_cls[reconstruct_config['precision']]

            if reconstruction_mode == 'separated':
                replacement_encoder_weight, replacement_decoder_weight = get_replacement_module(
                    weight=target.weight.T,
                    module_name=key,
                    type=reconstr_type,
                    reconstruct_config=reconstruct_config
                )

                if not isinstance(target, peft.tuners.lora.Linear):
                    raise NotImplementedError('Only initialization for peft.tuners.lora.Linear type is implemented.')
                    # TODO implement for Linear8bitLt
                else:
                    if half_init_dec:
                        kaiming_uniform_init_lower_half(replacement_decoder_weight)
                    if replacement_module_random_init:
                        kaiming_uniform_init(replacement_encoder_weight)
                        kaiming_uniform_init(replacement_decoder_weight)
                    replace_module_weights(target.lora_B.default, replacement_decoder_weight.T)
                    assert r_squared == True, "r_squared should be set"
                    target.forward = types.MethodType(lora_forward_latent, target)
                    target.get_delta_weight = types.MethodType(lora_get_delta_weight_faster_chatgpt, target)
                    replace_module_weights(target.lora_A.default, replacement_encoder_weight.T)

                    if learn_diagonal_only:
                        assert reconstruct_config['r_trainable'] is not None, "r_trainable should be set for learn_diagonal_only"
                        target.default_lora_latent_mapping = RandomLinear(lora_config.r, reconstruct_config['r_trainable'])
                        # init_module_weights(target.default_lora_latent_mapping.weight, sigma=0.00001)
                        torch.nn.init.normal_(target.default_lora_latent_mapping.weight, mean=0, std=0.00001)
                    else:
                        target.default_lora_latent_mapping = LINEAR_MODULE(lora_config.r, lora_config.r, bias=False)
                        init_module_weights(target.default_lora_latent_mapping, sigma=0.00001)
                    # Mark this module to prevent FSDP wrapping which can cause bias issues
                    target.default_lora_latent_mapping._is_lora_latent_mapping = True
                    target.default_lora_latent_mapping.to(target.lora_A.default.weight.device)

                    all_linear_names.append(key)
                    all_linears.append(target.default_lora_latent_mapping)

                    target.lora_A.default.weight.requires_grad = False  # only the r*r matrix will be tuned
                    target.lora_B.default.weight.requires_grad = False  # only the r*r matrix will be tuned

                pbar.set_postfix(num_lora_xs_modules=num_lora_xs_modules)

            else:
                raise NotImplementedError("The only supported mode is: separated.")
    
    def tie_linears(linear_layers: list[torch.nn.Linear], N_to_tie: int):
        i = 0
        while i < len(linear_layers):
            for j in range(i+1, i+N_to_tie):
                if j >= len(linear_layers):
                    break
                linear_layers[j].weight = linear_layers[i].weight
                print(f"[lora_xs.find_and_initialize] Tied {j} to layer {i}.")
            i += N_to_tie

    if reconstruct_config.get('tie_linear_num') > 0:
        i = 0
        N_to_tie = reconstruct_config.get('tie_linear_num')
        if reconstruct_config.get('tie_linear_mode') == 'tiled':
            # tiled weight-tying
            tie_linears(all_linears, N_to_tie)
        else:
            # structured weight-tying
            weight_type_keys = {}
            for i, (linear, name) in enumerate(zip(all_linears, all_linear_names)):
                try:
                    # weight_type = re.search(r'\d+.(\w+\.\w+)\.', name).group(1)
                    weight_type = re.search(r'\d+.(.+\..+).?.*', name).group(1)
                except:
                    weight_type = name
                    raise ValueError(f"Failed to extract weight type from {name}")
                if weight_type not in weight_type_keys:
                    weight_type_keys[weight_type] = []
                weight_type_keys[weight_type].append(linear)
            print(f"[lora_xs.find_and_initialize] Found {len(weight_type_keys)} weight types.")
            for weight_type, linear_layers in weight_type_keys.items():
                print(f"[lora_xs.find_and_initialize] Tying {weight_type} layers: {len(linear_layers)}")
                tie_linears(linear_layers, N_to_tie)
            # TODO: Consider how to handle off-multiples.
        # Count unique weight matrices by memory address
        unique_data_ptrs = set()
        for linear in all_linears:
            unique_data_ptrs.add(linear.weight.data.data_ptr())
        num_unique_matrices = len(unique_data_ptrs)
        dtype = all_linears[0].weight.dtype
        device = all_linears[0].weight.device
        print(f"[lora_xs.find_and_initialize] Checksum - number of unique weight matrices: {num_unique_matrices}, dtype: {dtype}, device: {device}")
        print(f"[lora_xs.find_and_initialize] Checksum - number of unique weight matrices: {num_unique_matrices}")

    if num_lora_xs_modules == 0:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )
