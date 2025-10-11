# https://github.com/MohammadrezaBanaei/LoRA-XS/blob/main/utils/initialization_utils.py

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
import torch.nn as nn
from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD
import numpy as np
from typing import Tuple

import torch.nn.functional as F

class DiagonalLinear(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.weight = torch.nn.Parameter(torch.randn(features))  # Renamed from diagonal
    
    def forward(self, x):
        return x * self.weight

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def get_delta_weight(self, adapter) -> torch.Tensor:
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
    if default_lora_latent_mapping.weight.dim() == 2:
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


def forward_latent(self, x: torch.Tensor):
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
        if latent_weight.dim() == 2:
            x = F.linear(x, latent_weight)
        elif latent_weight.dim() == 1:
            x = x * latent_weight  # element-wise multiplication for 1D case
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

def run_svd(input_matrix: np.ndarray, rank: int, n_iter: int, random_state: int) -> Tuple[np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=rank, n_iter=n_iter, random_state=random_state)
    svd.fit(input_matrix)
    reduced_matrix = svd.transform(input_matrix)
    return reduced_matrix, svd


def get_linear_rec_svd(input_matrix: np.ndarray, rank: int, n_iter: int,
                       random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced_matrix, svd = run_svd(input_matrix, rank, n_iter, random_state)

    reconstructed_matrix = svd.inverse_transform(reduced_matrix)
    return reconstructed_matrix, reduced_matrix, svd.components_


def get_replacement_module(weight, module_name, type, reconstruct_config):
    cfg = reconstruct_config[type]
    if type == 'svd':
        reconstructed_matrix, enc, dec = get_linear_rec_svd(
            input_matrix=weight.float().cpu().detach().numpy(), 
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


def init_diagonal_weights(target_module: DiagonalLinear, sigma: float):
    # Initialize diagonal weights close to 1 (identity-like) for better performance
    # For diagonal matrices, we want values around 1, not 0
    torch.nn.init.normal_(target_module.weight, mean=1.0, std=sigma)


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

                # print('[a] replacement_weight shape and dtype', replacement_encoder_weight.shape, replacement_encoder_weight.dtype)
                # print('[b] replacement_weight shape and dtype', replacement_decoder_weight.shape, replacement_decoder_weight.dtype)
                
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
                    assert r_squared, f"r_squared must be True when using Lora-XS"
                    target.forward = types.MethodType(forward_latent, target)
                    target.get_delta_weight = types.MethodType(get_delta_weight, target)
                    replace_module_weights(target.lora_A.default, replacement_encoder_weight.T)

                    if learn_diagonal_only:
                        target.default_lora_latent_mapping = DiagonalLinear(lora_config.r)
                        init_diagonal_weights(target.default_lora_latent_mapping, sigma=0.00001)
                    else:
                        target.default_lora_latent_mapping = LINEAR_MODULE(lora_config.r, lora_config.r, bias=False)
                        init_module_weights(target.default_lora_latent_mapping, sigma=0.00001)
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

    # import pdb; pdb.set_trace()
    if num_lora_xs_modules == 0:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )


class RandomProjection(nn.Module):
    def __init__(self, projection_dim, total_params, dtype):
        super().__init__()
        self.v = nn.Parameter(torch.randn(projection_dim, dtype=dtype))
        self.projection = nn.Linear(projection_dim, total_params, bias=False, dtype=dtype) 
        self._dim = projection_dim
    
    def forward(self, _):
        return self.projection(self.v) / math.sqrt(self._dim)

class Slicer(nn.Module):
    def __init__(self, rp, offset, size, shape):
        super().__init__()
        self.rp = rp
        self.offset = offset
        self.size = size
        self.shape = shape
    
    def forward(self, _):
        return self.rp(None)[self.offset:self.offset+self.size].view(self.shape)


def project_trainable_parameters(model: torch.nn.Module, rank: int) -> None:        
    trainable_params = []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                trainable_params.append((module, name, param))

    total_params = sum(p.numel() for _, _, p in trainable_params)
    dtype = trainable_params[0][2].dtype

    rp = RandomProjection(
        projection_dim=rank, 
        total_params=total_params, 
        dtype=dtype,
        rank=rank
    )

    offset = 0
    for module, name, param in trainable_params:
        size = param.numel()
        shape = param.shape
        param.requires_grad = False
        torch.nn.utils.parametrize.register_parametrization(module, name, Slicer(rp, offset, size, shape).to(param.device))
        offset += size

    rp.v.requires_grad = True
    rp.projection.weight.requires_grad = False

    # init_module_weights(rp.projection, sigma=0.00001)
    # init_module_weights(rp.v, sigma=0.00001)

    new_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

    print(f"Created RandomProjection with projection_dim={rank}, before_trainable_params={total_params}, after_trainable_params={new_total_trainable_params}, dtype={dtype}")
    # print(f"Projected {total_params} trainable parameters to {rank} dimensions.")