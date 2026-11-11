"""Each fine-tuning strategy must train exactly the parts it claims to, and leave the rest untouched.

Several of these fail today because of known bugs; the goals.md ID is noted on each such test.
"""

import pytest
import torch
from conftest import (
    MODEL_NAMES,
    apply_strategy,
    head_parameters,
    snapshot_buffers,
    snapshot_parameters,
    train_one_epoch,
)

from finetuning.FineTuneStrategy import FineTuneStrategy

pytestmark = pytest.mark.parametrize("name", MODEL_NAMES)


def _changed(before: dict[str, torch.Tensor], model: torch.nn.Module) -> set[str]:
    return {n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())}


def _names_of(model: torch.nn.Module, params: set) -> set[str]:
    return {n for n, p in model.named_parameters() if p in params}


# --------------------------------------------------------------------------- FULL_FT


def test_full_ft_trains_everything(name, make_model, random_loader):
    model = make_model(name)
    apply_strategy(model, FineTuneStrategy.FULL_FT)

    assert all(p.requires_grad for p in model.parameters())

    before = snapshot_parameters(model)
    train_one_epoch(model, random_loader)

    unchanged = set(before) - _changed(before, model)
    assert not unchanged, f"parameters not updated under FULL_FT: {sorted(unchanged)}"


# --------------------------------------------------------------------------- LINEAR_PROBE


def test_linear_probe_only_head_trainable(name, make_model):
    model = make_model(name)
    head = head_parameters(model)
    apply_strategy(model, FineTuneStrategy.LINEAR_PROBE)

    trainable = {p for p in model.parameters() if p.requires_grad}
    assert trainable == head


def test_linear_probe_freezes_backbone_weights(name, make_model, random_loader):
    model = make_model(name)
    head_names = _names_of(model, head_parameters(model))
    apply_strategy(model, FineTuneStrategy.LINEAR_PROBE)

    before = snapshot_parameters(model)
    train_one_epoch(model, random_loader)
    changed = _changed(before, model)

    assert changed == head_names


def test_linear_probe_keeps_batchnorm_stats(name, make_model, random_loader):
    """C4: `_train_epoch` calls model.train() on the frozen backbone, so BatchNorm stats drift."""
    model = make_model(name)
    apply_strategy(model, FineTuneStrategy.LINEAR_PROBE)

    before = snapshot_buffers(model)
    train_one_epoch(model, random_loader)

    changed = sorted(n for n, b in model.named_buffers() if not torch.equal(before[n], b))
    assert not changed, f"{len(changed)} frozen buffers changed, e.g. {changed[:3]}"


# --------------------------------------------------------------------------- LORA


def test_lora_targets_match_modules(name, make_model):
    """C2 / C3: a target pattern that matches no module is silently ignored by PEFT."""
    model = make_model(name)
    module_names = [n for n, _ in model.base_model.named_modules()]

    unmatched = [
        target
        for target in model.adapter.get_lora_target_modules()
        if not any(n == target or n.endswith("." + target) for n in module_names)
    ]
    assert not unmatched, f"LoRA targets matching no module: {unmatched}"


def test_lora_trainable_params_are_lora_and_head(name, make_model):
    """C1: PEFT freezes everything but lora_ params, and nothing re-enables the head."""
    model = make_model(name)
    head = head_parameters(model)
    apply_strategy(model, FineTuneStrategy.LORA)

    lora = {p for n, p in model.named_parameters() if "lora_" in n}
    trainable = {p for p in model.parameters() if p.requires_grad}

    assert lora, "no LoRA parameters were created"
    assert trainable == lora | head, (
        f"missing from trainable: {sorted(_names_of(model, (lora | head) - trainable))}; "
        f"unexpectedly trainable: {sorted(_names_of(model, trainable - (lora | head)))}"
    )


def test_lora_freezes_base_weights(name, make_model, random_loader):
    model = make_model(name)
    head = head_parameters(model)
    apply_strategy(model, FineTuneStrategy.LORA)
    base_names = {n for n, p in model.named_parameters() if "lora_" not in n and p not in head}

    before = snapshot_parameters(model)
    train_one_epoch(model, random_loader)

    changed_base = _changed(before, model) & base_names
    assert not changed_base, f"base weights changed under LoRA: {sorted(changed_base)}"


def test_lora_adapters_receive_updates(name, make_model, random_loader):
    """C2 / C3: LoRA wrapped around a module whose forward is never called gets no gradient.

    Only lora_B is checked: it starts at zero, so lora_A gets no gradient on the first step.
    """
    model = make_model(name)
    apply_strategy(model, FineTuneStrategy.LORA)
    lora_b = {n for n, _ in model.named_parameters() if "lora_B" in n}
    assert lora_b, "no LoRA parameters were created"

    before = snapshot_parameters(model)
    train_one_epoch(model, random_loader)

    stale = sorted(lora_b - _changed(before, model))
    assert not stale, f"{len(stale)}/{len(lora_b)} lora_B weights never updated, e.g. {stale[:3]}"


def test_lora_head_is_trained(name, make_model, random_loader):
    """C1: the new classification head must learn under LoRA, not stay at its random init."""
    model = make_model(name)
    head = head_parameters(model)
    apply_strategy(model, FineTuneStrategy.LORA)
    head_names = _names_of(model, head)  # names after PEFT's renaming

    before = snapshot_parameters(model)
    train_one_epoch(model, random_loader)

    stale = sorted(head_names - _changed(before, model))
    assert not stale, f"head parameters not updated under LoRA: {stale}"
