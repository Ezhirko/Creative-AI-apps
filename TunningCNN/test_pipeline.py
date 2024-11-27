import pytest
import torch.nn as nn
from SimpleCNN18K import SimpleCNN18K

@pytest.fixture
def model():
    """Fixture to provide the model instance."""
    return SimpleCNN18K()

def test_total_parameters(model):
    """Test that the model has fewer than 100,000 parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, exceeding the limit."

def test_use_of_batch_normalization(model):
    """Test that the model uses at least one Batch Normalization layer."""
    has_batch_norm = any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
    assert has_batch_norm, "Model does not use Batch Normalization."

def test_use_of_dropout(model):
    """Test that the model uses at least one Dropout layer."""
    has_dropout = any(isinstance(layer, (nn.Dropout, nn.Dropout2d)) for layer in model.modules())
    assert has_dropout, "Model does not use Dropout."

def test_use_of_gap_layer(model):
    """Test that the model uses a Global Average Pooling (GAP) layer."""
    has_gap = any(isinstance(layer, nn.AdaptiveAvgPool2d) for layer in model.modules())
    assert has_gap, "Model does not use a Global Average Pooling (GAP) layer."

# Run tests only if the script is executed directly (optional with pytest)
if __name__ == "__main__":
    pytest.main()