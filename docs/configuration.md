# Configuration

What a user sets. The Axolotl plugin exports what the kernels need, so most runs set only the
config keys below. The environment variables restore a previous default or tune memory.

## Axolotl config keys

| Key | Values | Meaning |
|---|---|---|
| `expert_parallel_size` | `1` (off) or a divisor of the world size | Number of EP ranks. Experts are split evenly across them. |
| `expert_parallel_backend` | `auto`, `deep_ep`, `torch`, `optmoe` | Token dispatch backend. `auto` never picks `optmoe`, so set it explicitly. |
| `expert_parallel_optmoe_mode` | `auto` (default), `fused`, `dispatch` | `fused`: one persistent kernel per direction, for plain SwiGLU experts with full fine-tuning or an expert LoRA adapter. `dispatch`: OPTMoE's dispatch and combine around the configured local experts kernel, for everything the fused mode does not cover. `auto` picks `fused` when the config and model allow it. |
| `expert_parallel_optmoe_precision` | `auto`, `bf16`, `mxfp8`, `fp8`, `nvfp4_lora`, `bf16_lora`, `mxfp8_lora`, `fp8_lora` | Compute precision of the fused kernels (next table). Master weights keep their dtype and gradients come back in it. |
| `expert_parallel_optmoe_fused_lora` | `true` (default), `false` | Train an expert LoRA adapter on the fused kernels. `false` uses the dispatch mode for every adapter except `nvfp4_lora`. |
| `expert_parallel_optmoe_lora_merged_cache`, `expert_parallel_optmoe_nvfp4_merged_cache` | `layer` (default), `all` | How long a layer's merged weights (base + LoRA) stay resident. `layer` builds them per layer and keeps the two most recent layers. `all` keeps every layer for the step: faster, more memory. |
| `expert_parallel_optmoe_nvfp4_headroom_log2` | `0` (default) to `4` | `nvfp4_lora` only: headroom of the merged NVFP4 weights over the base expert's largest magnitude, as a power of two. `0` clamps merged weights at the base's largest magnitude. Exported as `OPTMOE_NVFP4_LORA_HEADROOM_LOG2`. |
| `expert_parallel_optmoe_workspace_bound` | `auto` (default), `max` | Size the backward pools and rings from each step's routed rows. `max` restores worst-case sizing. |
| `expert_parallel_optmoe_compact`, `expert_parallel_optmoe_bounded` | `false` (default), `true` | Dispatch mode only: hand the local kernel only the routed rows. `bounded` avoids the host read that `compact` adds. |
| `expert_parallel_fallback_on_unsupported` | `true` (default), `false` | When `optmoe` cannot run, fall back to `expert_parallel_backend: torch` with a warning, or raise. |

Required alongside: `expert_parallel_size x dp_shard_size` equal to the world size with the EP
group inside one NVLink node, non-reentrant `activation_checkpointing`, and
`fsdp_config.state_dict_type: FULL_STATE_DICT`.

## Precisions

| Precision | SM90 (H100, H200) | SM100 (B200) | Notes |
|---|---|---|---|
| `auto` | `bf16` | `mxfp8` | With an expert LoRA adapter, resolves to the matching `*_lora` precision. With packed NVFP4 experts and LoRA on SM100, resolves to `nvfp4_lora`. |
| `bf16` | yes | yes | About 0.2 % relative error against an FP32 reference. |
| `mxfp8` | no | yes | MXFP8 forward and backward. About 6-7 % relative error against FP32. Hidden sizes that are not multiples of 512 take the BF16 backward, logged once. |
| `fp8` | yes | no | Blockwise E4M3 forward and backward. |
| `bf16_lora`, `mxfp8_lora`, `fp8_lora` | as the base precision | as the base precision | Frozen BF16 experts with LoRA adapters on the full fine-tune kernels. |
| `nvfp4_lora` | no | yes | Frozen packed NVFP4 experts (modelopt checkpoint) with BF16 LoRA adapters. Hidden and intermediate sizes must be multiples of 256. |

The fused LoRA precisions need `adapter: lora` with `lora_target_parameters` on both
`experts.gate_up_proj` and `experts.down_proj`, `use_sonicmoe: true`, `lora_dropout: 0` and no
DoRA. `nvfp4_lora` also needs `nvfp4_merge_aware: false`. Only the adapter is saved. The package
API for these paths is in [trl-integration.md](trl-integration.md#nvfp4-lora).

## When the backend falls back

The fused mode needs all of the following. Otherwise `auto` uses the dispatch mode where it
can, and the backend falls back to `torch` or raises, per
`expert_parallel_fallback_on_unsupported`.

- The `optmoe` package with its DeepGEMM build, on SM90 or SM100 GPUs.
- `hidden_size % 256 == 0` and expert intermediate size `% 128 == 0`.
- Experts in the HF `gate_up_proj [E, 2I, H]` / `down_proj [E, H, I]` layout, the same shape in
  every MoE layer.
- Plain SwiGLU experts: no biases, no clamp, no custom gate.
- No quantized loading (`load_in_4bit`, `load_in_8bit`, `gptq`, `model_quantization_config`),
  apart from the NVFP4 checkpoint that `nvfp4_lora` trains on.
- A precision the GPU architecture supports, and an adapter config that qualifies for a fused
  LoRA precision when an adapter is set.

## FSDP2 and `NCCL_MAX_CTAS`

The persistent kernels occupy every SM they are given, so FSDP's NCCL collectives must fit in
the SMs the runtime reserves (`reserved_sms`, default 16), or the two deadlock. `NCCL_MAX_CTAS`
must be `reserved_sms // 2` or lower (`// 3` under HSDP) before the process group exists. The
Axolotl plugin exports it. Outside Axolotl, export `optmoe.nccl_env(reserved_sms, dp_replicate)`
yourself; `check_nccl_env` raises early when it is missing or too large. Raise `reserved_sms`
if collectives still stall: the four-GPU `dp_shard 2 x EP 2` mesh on B200 needed 24.

## Environment variables

Set these in the environment of every rank.

| Variable | Default | Other values | Effect |
|---|---|---|---|
| `DG_JIT_CACHE_DIR` | `~/.dj` | a shared, persistent directory | Where compiled kernel variants are cached. |
| `NCCL_MAX_CTAS` | exported by the plugin | | See above. |
| `OPTMOE_WEIGHT_CACHE` | `1` | `0` | Cache the per-step weight layouts. `0` rebuilds them on every call. |
| `OPTMOE_WORKSPACE_BOUND` | `auto` | `max` | Same as `expert_parallel_optmoe_workspace_bound`. |
| `OPTMOE_LORA_MERGED_CACHE`, `OPTMOE_NVFP4_MERGED_CACHE` | `layer` | `all` | Same as the config keys of the same name. |
| `OPTMOE_STORED_CONTEXT` | `auto` | `off` | SM100 BF16: reuse the forward's gate/up pre-activations in the backward instead of recomputing them. |
| `OPTMOE_SM100_BACKWARD_PRECISION` | `mxfp8` under `precision: mxfp8`, else `bf16` | `bf16`, `mxfp8` | Precision of the SM100 backward GEMMs. |
| `OPTMOE_SM100_DW_PRECISION` | `mxfp8` under `precision: mxfp8`, else `bf16` | `bf16`, `mxfp8` | Precision of the SM100 weight gradients. `bf16` costs more time and memory. |
| `OPTMOE_SM100_MXFP8_WEIGHT_FORMS` | `cuda` | `fused`, `compiled`, `triton` | Builder of the MXFP8 backward weight forms. All four give identical forms. |
| `OPTMOE_SM90_FP8_TRANSPORT`, `OPTMOE_SM90_FP8_COMBINE` | `1` | `0` | SM90 `fp8` / `fp8_lora` backward: E4M3 transport and combine. `0` restores BF16. |
| `OPTMOE_NVFP4_REQUANT` | `cuda_fused` | `cuda`, `triton` | Kernel that rebuilds the merged NVFP4 weights each step. |
| `OPTMOE_NVFP4_LORA_HEADROOM_LOG2` | `0` | `1` to `4` | Headroom of the merged NVFP4 weights over the base's largest magnitude, as a power of two. |
| `OPTMOE_NVFP4_BACKWARD_RING_WAVES` | `1` | `0` | Ring floor of the NVFP4 LoRA backward. |
| `OPTMOE_SM100_NVFP4_BACKWARD_OPTIONS` | `2051` | `3`, `0` | NVFP4 LoRA backward option bits. |

The remaining `OPTMOE_*` variables in `src/optmoe` are kernel tuning selectors. Their defaults
are the measured ones and they are not meant to be set for training. Changing a kernel selector
compiles a new kernel variant.

## Troubleshooting

- **`TypeError: incompatible function arguments` from `optmoe_deepgemm._C`.** The installed extension
  is older than the Python package. Reinstall it with `python -m optmoe.install_kernels`
  (from a checkout: `--source .`, which builds clean).
- **The first step takes minutes.** Every kernel variant compiles on first use. Point
  `DG_JIT_CACHE_DIR` at a persistent, shared directory so later runs and other ranks reuse the
  builds.
- **A hang, then `DeepGEMM grid sync timeout` or `DeepGEMM combine peers timeout`.** A NCCL
  collective needs more CTAs than the reserved SMs hold. See the FSDP2 section.
- **`expert_parallel_backend=optmoe is unsupported here: <reason>`.** The model or config hits
  a fallback condition. Set `expert_parallel_optmoe_mode: dispatch` when only the fused mode is
  excluded, or `expert_parallel_fallback_on_unsupported: false` to raise instead.
