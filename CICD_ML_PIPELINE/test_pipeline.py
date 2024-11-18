import pytest
import torch
from train import SimpleCNN
import train as trn

@pytest.fixture
def model():
    """Fixture to provide the model instance."""
    return SimpleCNN()

def test_total_parameters(model):
    """Test that the model has fewer than 100,000 parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000, f"Model has {total_params} parameters, exceeding the limit."

def test_input_compatibility(model):
    """Test that the model accepts 28x28 input without errors."""
    sample_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(sample_input)
    except Exception as e:
        pytest.fail(f"Model failed with 28x28 input: {e}")
    
def test_output_shape(model):
    """Test that the model output has 10 units."""
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    assert output.shape[1] == 10, f"Model output shape mismatch: {output.shape}"

def test_accuracy_greater_than_95():
    accuracy = trn.train()
    assert accuracy >= 95, f"Model has {accuracy} accuracy, not meeting the requirement."

# Run tests only if the script is executed directly (optional with pytest)
if __name__ == "__main__":
    pytest.main()
