"""Fine-tuning the coach.

DPO is the primary objective because the thing we want to teach — reflective
over directive — is a preference between two plausible answers, not a single
correct string. SFT on the ``chosen`` side is offered as a fallback for
environments where DPO's reference-model memory overhead doesn't fit.

The notebook version wrapped training in a bare ``except`` that swallowed the
error and silently fell through to a second trainer. Here the fallback is an
explicit choice (``--strategy sft``, or ``--fallback-to-sft``), and failures are
logged with their traceback before anything else happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aura.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Defaults tuned for a single 16 GB GPU (a Colab T4 or similar)."""

    model_id: str = "unsloth/gemma-3n-E2B-it"
    output_dir: Path = Path("artifacts/wellness-coach")
    strategy: str = "dpo"  # "dpo" | "sft"
    fallback_to_sft: bool = False

    max_seq_length: int = 2048
    load_in_4bit: bool = True

    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    finetune_vision_layers: bool = False

    max_steps: int = 200
    num_train_epochs: float | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 50
    beta: float = 0.1

    psychology_limit: int | None = 3000
    eval_fraction: float = 0.05
    seed: int = 3407
    push_to_hub: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)


def load_model(config: TrainingConfig) -> tuple[Any, Any]:
    """Load the base model with LoRA adapters attached."""
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    log.info("loading %s (4bit=%s)", config.model_id, config.load_in_4bit)
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.model_id,
        dtype=None,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=config.finetune_vision_layers,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    return model, tokenizer


def _precision_flags() -> dict[str, bool]:
    """bf16 where the hardware supports it; fp16 otherwise."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return {"bf16": True, "fp16": False}
    except Exception:  # pragma: no cover
        pass
    return {"bf16": False, "fp16": True}


def train_dpo(model: Any, tokenizer: Any, train_ds: Any, eval_ds: Any, config: TrainingConfig):
    from trl import DPOConfig, DPOTrainer

    args = DPOConfig(
        output_dir=str(config.output_dir / "dpo"),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps if config.num_train_epochs is None else -1,
        num_train_epochs=config.num_train_epochs or 1.0,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        beta=config.beta,
        loss_type="sigmoid",
        seed=config.seed,
        report_to="none",
        remove_unused_columns=False,
        # Single-threaded loading: the multimodal processor is not fork-safe.
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=config.save_steps if eval_ds is not None else None,
        **_precision_flags(),
    )
    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
    )
    log.info("starting DPO on %d pairs", len(train_ds))
    return trainer.train()


def train_sft(model: Any, tokenizer: Any, train_ds: Any, config: TrainingConfig):
    from trl import SFTConfig, SFTTrainer

    from aura.training.data import to_sft_dataset

    dataset = to_sft_dataset(train_ds, tokenizer)
    args = SFTConfig(
        output_dir=str(config.output_dir / "sft"),
        dataset_text_field="text",
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps if config.num_train_epochs is None else -1,
        num_train_epochs=config.num_train_epochs or 1.0,
        learning_rate=max(config.learning_rate, 2e-5),
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        gradient_checkpointing=True,
        seed=config.seed,
        report_to="none",
        dataloader_num_workers=0,
        **_precision_flags(),
    )
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset, args=args
    )
    log.info("starting SFT on %d examples", len(dataset))
    return trainer.train()


def run(config: TrainingConfig) -> Path:
    """Build the dataset, train, and save the adapter. Returns the save path."""
    from aura.training.data import build_preference_dataset

    train_ds, eval_ds = build_preference_dataset(
        psychology_limit=config.psychology_limit,
        seed=config.seed,
        eval_fraction=config.eval_fraction,
    )
    model, tokenizer = load_model(config)

    strategy = config.strategy
    try:
        stats = (
            train_dpo(model, tokenizer, train_ds, eval_ds, config)
            if strategy == "dpo"
            else train_sft(model, tokenizer, train_ds, config)
        )
    except Exception:
        log.exception("%s training failed", strategy.upper())
        if not (strategy == "dpo" and config.fallback_to_sft):
            raise
        log.warning("falling back to SFT as requested")
        strategy = "sft"
        stats = train_sft(model, tokenizer, train_ds, config)

    runtime = stats.metrics.get("train_runtime", 0) / 60
    log.info("%s finished in %.1f min — loss %.4f", strategy.upper(), runtime,
             stats.metrics.get("train_loss", float("nan")))

    destination = config.output_dir / "adapter"
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(destination))
    tokenizer.save_pretrained(str(destination))
    log.info("adapter saved to %s", destination)

    if config.push_to_hub:
        model.push_to_hub(config.push_to_hub)
        tokenizer.push_to_hub(config.push_to_hub)
        log.info("pushed to https://huggingface.co/%s", config.push_to_hub)

    return destination
