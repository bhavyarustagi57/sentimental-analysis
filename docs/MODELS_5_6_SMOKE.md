# Models 5–6 decoder/LoRA smoke validation

## Scope and designation

Only `google/gemma-3-270m` and `Qwen/Qwen3-0.6B-Base` were addressed. No final
benchmark, canonical result, live API routing, frontend, dataset membership, or
classical model was changed. All executed metrics below are **SMOKE TEST — NOT
FINAL BENCHMARK**.

## Immutable revisions

- Gemma model/tokenizer: `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- Qwen model/tokenizer: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`

The four existing encoder pins remain unchanged.

## Gemma access

Access was initially blocked by the repository's manual license gate. After the
user accepted the license and authenticated this machine, pinned `config.json`,
tokenizer, and model requests all succeeded. No credential was stored in the
repository.

## Gemma architecture and tokenizer inspection

The actual pinned checkpoint loaded natively as
`Gemma3TextForSequenceClassification` with 268,099,456 base-classifier
parameters. Its 18 layers each expose `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
its newly initialized binary head is `score`.

The tokenizer is `GemmaTokenizer`, vocabulary size 262,145, pad ID 0 and EOS ID
1. The upstream tokenizer loaded with left padding; the shared classifier policy
explicitly changed it to right padding. Dynamic padding produced matching
`(2, 10)` input/mask tensors with zero-valued padded mask positions. A 400-word
input truncated to exactly 256 tokens.

The downloaded `model.safetensors` is 536,223,056 bytes with SHA-256
`abb58eb73aece5163624090d5383c59cbcdb9e9539ec84cb864f74386c4b21d5`.

## Gemma LoRA configuration and real training

- Task: sequence classification, two logits
- Rank: 4
- Alpha: 8
- Dropout: 0.05
- Targets: `q_proj`, `v_proj`
- Modules to save: `score`
- Bias: none
- Trainable parameters: 185,600
- Total PEFT-wrapped parameters: 268,285,056
- Trainable percentage: 0.0691801484%
- Device/precision: CPU/FP32
- Dataset: IMDb, existing locked split manifest
- Balanced smoke sizes: 4 train / 2 validation / 2 test
- Physical batch size: 1
- Gradient accumulation: 1
- Optimizer steps: 2
- Effective epoch fraction: 0.5
- Optimizer: AdamW, learning rate 2e-4
- Training losses: 0.0847000777721405, 4.661614418029785
- Training time: 5.4573237000149675 seconds
- LoRA gradient seen and parameters updated: YES
- Score-head gradient seen and parameters updated: YES

Validation accuracy was 0.5, macro F1 0.3333333333333333, and loss
1.1414938782108948. Smoke-test accuracy was 0.5, macro F1
0.3333333333333333, and loss 0.40514428447932005. These tiny results do not
measure model quality and test data did not participate in checkpoint selection.

Gemma adapter persistence contains 73 tensors, including LoRA and `score`
tensors. `adapter_model.safetensors` is 751,784 bytes; the complete checkpoint is
758,147 bytes. It contains no full base checkpoint and no legacy `.bin` file.

For fixed text `A thoughtful and genuinely wonderful film.`, pre-save and
post-reload logits were both `[0.651352047920227, -2.393599033355713]`.
Maximum absolute difference was 0.0 at tolerance 1e-5.

Forced-offline reconstruction passed with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. It requires the immutable gated base snapshot to remain
available locally; the small PEFT artifact alone is not a standalone model.

## Qwen architecture and tokenizer inspection

The exact pinned checkpoint loaded as `Qwen3ForSequenceClassification` with
596,051,968 base-classifier parameters before PEFT wrapping. Its 28 layers each
expose `q_proj`, `k_proj`, `v_proj`, and `o_proj`; its newly initialized binary
head is `score`. The tokenizer is `Qwen2Tokenizer`, vocabulary size 151,669,
right-padding, with pad token equal to EOS (ID 151643).

Dynamic padding produced matching `(2, 9)` input/mask tensors with zero-valued
mask positions for padding. A 400-word input truncated to exactly 256 tokens.

The downloaded `model.safetensors` is 1,192,135,096 bytes with SHA-256
`cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`.

## Qwen LoRA configuration

- Task: sequence classification, two logits
- Rank: 4
- Alpha: 8
- Dropout: 0.05
- Targets: `q_proj`, `v_proj`
- Modules to save: `score`
- Bias: none
- Trainable parameters: 575,488
- Total PEFT-wrapped parameters: 596,627,456
- Trainable percentage: 0.0964568416%

The query/value-only target is the smallest conventional attention adaptation
set confirmed against the real loaded module graph. PEFT's serialized config may
also list its automatically recognized classifier aliases; safetensors inspection
is the authoritative persistence proof.

## Real Qwen training

- Device/precision: CPU/FP32
- Dataset: IMDb, existing locked split manifest
- Split hash: `fba6491bca30fa9dcf2a8c48a9d945b3d7042f98f5d61e3415e24664022cb327`
- Persisted split membership: 7,200 train / 800 validation / 2,000 test
- Balanced smoke sizes: 4 train / 2 validation / 2 test
- Physical batch size: 1
- Gradient accumulation: 1
- Optimizer steps: 2
- Effective epoch fraction: 0.5
- Optimizer: AdamW, learning rate 2e-4
- Training losses: 0.012123115360736847, 3.588867425918579
- Training time: 23.21649800002342 seconds
- LoRA gradient seen and parameters updated: YES
- Score-head gradient seen and parameters updated: YES

Validation accuracy was 0.5, macro F1 0.3333333333333333, and loss
1.5564880594611168. Smoke-test accuracy was 0.5, macro F1
0.3333333333333333, and loss 1.4221429601311684. These tiny results do not
measure model quality and test data did not participate in checkpoint selection.

## Persistence and reload

The artifact contains a base model ID and immutable revision, tokenizer, model
config, `adapter_config.json`, `adapter_model.safetensors`, classification head,
and metadata. It does not duplicate the full base model.

- Adapter safetensors: 2,316,720 bytes
- Complete checkpoint directory: 2,323,086 bytes
- Tensors: 113
- LoRA tensors present: YES
- Score tensors present: YES
- Full `model.safetensors` in adapter checkpoint: NO
- Legacy `.bin` files: NO

For fixed text `A thoughtful and genuinely wonderful film.`, pre-save and
post-reload logits were both `[-0.23958200216293335, -2.03389573097229]`.
Maximum absolute difference was 0.0 at tolerance 1e-5.

Forced-offline reconstruction passed with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. This honestly depends on the immutable Qwen base
snapshot remaining in the local Hugging Face cache; the 2.3 MB adapter alone is
not a standalone model.

## Foundation defects

1. The common loader reported hardware-selected FP32 but did not pass a dtype to
   Transformers, allowing Qwen's BF16 checkpoint dtype to override the plan. The
   loader now maps the resolved precision to an explicit torch dtype.
2. CPU execution enabled decoder gradient checkpointing intended for constrained
   GPUs, causing extreme recomputation without a relevant CPU-memory benefit.
   Checkpointing now activates only when the registry requests it and the resolved
   device is CUDA. The GPU policy is unchanged.
3. The first integrity comparison converted trainable tensors through NumPy,
   which does not support PyTorch BF16. It now compares tensors directly using
   `torch.equal` and is dtype-safe.

The failed pre-correction Qwen attempt completed optimizer work but wrote no
artifact and was not promoted as evidence. The clean FP32 rerun is the recorded
result.

## GPU status

- CUDA execution: NOT VALIDATED
- FP16/BF16: NOT VALIDATED
- Real GPU OOM recovery: NOT VALIDATED
