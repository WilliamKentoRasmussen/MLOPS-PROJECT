"""39 Unit tests for the model module."""

import pytest
import torch
from main_project.model import BaselineCNN, get_model
from torch import nn


class TestBaselineCNN:
    """Test suite for BaselineCNN model."""

    @pytest.mark.parametrize("num_classes", [2, 3, 10])
    def test_baseline_initialization(self, num_classes):
        """Test that BaselineCNN initializes correctly with different num_classes."""
        model = BaselineCNN(num_classes=num_classes)
        assert isinstance(model, nn.Module), "Model should be an instance of nn.Module"
        assert hasattr(model, "features"), "Model should have 'features' attribute"
        assert hasattr(model, "classifier"), "Model should have 'classifier' attribute"

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_baseline_forward_pass(self, batch_size):
        """Test forward pass with different batch sizes."""
        model = BaselineCNN(num_classes=2)
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)

        expected_shape = (batch_size, 2)
        assert output.shape == expected_shape, f"Output shape should be {expected_shape}, got {output.shape}"

    def test_baseline_output_shape(self):
        """Test that output has correct shape for binary classification."""
        model = BaselineCNN(num_classes=2)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)

        assert output.shape == (1, 2), f"Output shape should be (1, 2) for binary classification, got {output.shape}"

    def test_baseline_input_validation(self):
        """Test that model raises error for incorrect input shape."""
        model = BaselineCNN(num_classes=2)

        # Wrong number of channels
        with pytest.raises(RuntimeError):
            x = torch.randn(1, 1, 224, 224)  # 1 channel instead of 3
            _ = model(x)

    @pytest.mark.parametrize("input_size", [(224, 224), (256, 256), (128, 128)])
    def test_baseline_different_input_sizes(self, input_size):
        """Test that model works with different input sizes."""
        model = BaselineCNN(num_classes=2)
        h, w = input_size
        x = torch.randn(1, 3, h, w)
        output = model(x)

        assert output.shape == (1, 2), f"Output shape should be (1, 2) regardless of input size, got {output.shape}"

    def test_baseline_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = BaselineCNN(num_classes=2)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None, "Gradients should flow back to input"

        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"


class TestGetModel:
    """Test suite for get_model factory function."""

    @pytest.mark.parametrize("model_name", ["baseline", "alexnet", "vgg16"])
    def test_get_model_returns_module(self, model_name):
        """Test that get_model returns a valid nn.Module for all model types."""
        model = get_model(model_name, pretrained=False)
        assert isinstance(model, nn.Module), f"get_model('{model_name}') should return an nn.Module instance"

    @pytest.mark.parametrize(
        "model_name,batch_size",
        [
            ("baseline", 1),
            ("baseline", 8),
            ("alexnet", 1),
            ("alexnet", 4),
            ("vgg16", 1),
            ("vgg16", 2),
        ],
    )
    def test_get_model_output_shape(self, model_name, batch_size):
        """Test that all models produce correct output shape."""
        model = get_model(model_name, num_classes=2, pretrained=False)
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)

        expected_shape = (batch_size, 2)
        assert (
            output.shape == expected_shape
        ), f"Model '{model_name}' output shape should be {expected_shape}, got {output.shape}"

    @pytest.mark.parametrize("num_classes", [2, 5, 10])
    def test_get_model_num_classes(self, num_classes):
        """Test that models can be created with different number of classes."""
        model = get_model("baseline", num_classes=num_classes, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)

        assert output.shape[1] == num_classes, f"Output should have {num_classes} classes, got {output.shape[1]}"

    def test_get_model_invalid_name(self):
        """Test that get_model raises ValueError for invalid model name."""
        with pytest.raises(ValueError, match="Unknown model"):
            _ = get_model("invalid_model_name")

    @pytest.mark.parametrize("model_name", ["baseline", "alexnet", "vgg16"])
    def test_get_model_pretrained_flag(self, model_name):
        """Test that pretrained flag works without errors."""
        # Test with pretrained=False
        model_no_pretrain = get_model(model_name, pretrained=False)
        assert isinstance(model_no_pretrain, nn.Module), f"Model '{model_name}' with pretrained=False should be valid"

        # Test with pretrained=True (only for alexnet and vgg16)
        if model_name != "baseline":
            model_pretrained = get_model(model_name, pretrained=True)
            assert isinstance(model_pretrained, nn.Module), f"Model '{model_name}' with pretrained=True should be valid"

    def test_get_model_parameter_count(self):
        """Test that models have reasonable number of parameters."""
        models_to_test = {
            "baseline": (100_000, 1_000_000),  # ~100K-1M params
            "alexnet": (50_000_000, 70_000_000),  # ~61M params
            "vgg16": (130_000_000, 150_000_000),  # ~138M params
        }

        for model_name, (min_params, max_params) in models_to_test.items():
            model = get_model(model_name, pretrained=False)
            num_params = sum(p.numel() for p in model.parameters())

            assert (
                min_params <= num_params <= max_params
            ), f"Model '{model_name}' should have between {min_params:,} and {max_params:,} parameters, got {num_params:,}"

    @pytest.mark.parametrize("model_name", ["baseline", "alexnet", "vgg16"])
    def test_model_training_mode(self, model_name):
        """Test that models can switch between train and eval modes."""
        model = get_model(model_name, pretrained=False)

        # Test training mode
        model.train()
        assert model.training, f"Model '{model_name}' should be in training mode"

        # Test eval mode
        model.eval()
        assert not model.training, f"Model '{model_name}' should be in eval mode"

    @pytest.mark.parametrize("model_name", ["baseline", "alexnet", "vgg16"])
    def test_model_device_transfer(self, model_name):
        """Test that models can be moved to different devices."""
        model = get_model(model_name, pretrained=False)

        # Test CPU
        model_cpu = model.to("cpu")
        x_cpu = torch.randn(1, 3, 224, 224)
        output_cpu = model_cpu(x_cpu)
        assert output_cpu.device.type == "cpu", f"Model '{model_name}' output should be on CPU"

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to("cuda")
            x_cuda = torch.randn(1, 3, 224, 224).to("cuda")
            output_cuda = model_cuda(x_cuda)
            assert output_cuda.device.type == "cuda", f"Model '{model_name}' output should be on CUDA"

    @pytest.mark.parametrize("model_name", ["baseline", "alexnet", "vgg16"])
    def test_model_deterministic_output(self, model_name):
        """Test that models produce deterministic output in eval mode."""
        torch.manual_seed(42)
        model = get_model(model_name, pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(
            output1, output2
        ), f"Model '{model_name}' should produce deterministic output in eval mode"

    def test_model_output_range(self):
        """Test that model outputs are in reasonable range (logits)."""
        model = get_model("baseline", pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        output = model(x)

        # Logits should typically be in a reasonable range
        assert output.min() > -100, "Output logits seem unreasonably small"
        assert output.max() < 100, "Output logits seem unreasonably large"
