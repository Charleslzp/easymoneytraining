"""
Exponential Moving Average (EMA) for model parameters
用于平滑模型参数，提高泛化性能和稳定性
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class EMA:
    """
    Exponential Moving Average wrapper for PyTorch models

    Usage:
        model = MyModel()
        ema = EMA(model, decay=0.999)

        # Training loop
        for batch in dataloader:
            loss = train_step(model, batch)
            optimizer.step()
            ema.update()  # Update EMA parameters

        # Evaluation with EMA
        ema.apply_shadow()
        eval_result = evaluate(model)
        ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: PyTorch model to track
            decay: EMA decay rate (0.999-0.9999 recommended)
                   Higher = smoother but slower to adapt
        """
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update EMA parameters after each training step
        Call this after optimizer.step()
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow, f"Parameter {name} not in shadow dict"
                    new_average = (
                        self.decay * self.shadow[name] +
                        (1.0 - self.decay) * param.data
                    )
                    self.shadow[name] = new_average

    def apply_shadow(self):
        """
        Replace model parameters with EMA parameters
        Call this before evaluation/inference
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """
        Restore original model parameters
        Call this after evaluation to continue training
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> Dict:
        """Get EMA state for checkpointing"""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load EMA state from checkpoint"""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
