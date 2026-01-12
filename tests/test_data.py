"""Unit tests for the data module."""

from pathlib import Path
import tempfile
import shutil
import os

import pytest
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from main_project.data import ChestXRayDataset

# Test configuration
_TEST_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")


@pytest.fixture
def mock_data_dir():
    """Create a temporary directory with mock chest X-ray data."""
    temp_dir = tempfile.mkdtemp()
    data_path = Path(temp_dir)
    
    # Create directory structure for train split
    train_dir = data_path / "train"
    normal_dir = train_dir / "NORMAL"
    pneumonia_dir = train_dir / "PNEUMONIA"
    
    normal_dir.mkdir(parents=True, exist_ok=True)
    pneumonia_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock images (100x100 grayscale images)
    for i in range(5):
        # Create NORMAL images
        img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode='L')
        img.save(normal_dir / f"normal_{i}.jpeg")
        
        # Create PNEUMONIA images
        img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode='L')
        img.save(pneumonia_dir / f"pneumonia_{i}.jpeg")
    
    yield data_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_data_dir_with_splits():
    """Create a temporary directory with mock data for all splits."""
    temp_dir = tempfile.mkdtemp()
    data_path = Path(temp_dir)
    
    for split in ["train", "val", "test"]:
        split_dir = data_path / split
        normal_dir = split_dir / "NORMAL"
        pneumonia_dir = split_dir / "PNEUMONIA"
        
        normal_dir.mkdir(parents=True, exist_ok=True)
        pneumonia_dir.mkdir(parents=True, exist_ok=True)
        
        # Create different number of images per split
        num_images = {"train": 10, "val": 3, "test": 2}[split]
        
        for i in range(num_images):
            img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode='L')
            img.save(normal_dir / f"normal_{i}.jpeg")
            
            img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode='L')
            img.save(pneumonia_dir / f"pneumonia_{i}.jpeg")
    
    yield data_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestChestXRayDataset:
    """Test suite for ChestXRayDataset class."""
    
    def test_dataset_is_instance_of_dataset(self, mock_data_dir):
        """Test that ChestXRayDataset is an instance of torch Dataset."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        assert isinstance(dataset, Dataset), "Dataset should be an instance of torch.utils.data.Dataset"
    
    def test_dataset_length(self, mock_data_dir):
        """Test __len__ method returns correct dataset size."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        expected_length = 10  # 5 NORMAL + 5 PNEUMONIA
        assert len(dataset) == expected_length, f"Dataset should have {expected_length} samples, got {len(dataset)}"
    
    @pytest.mark.parametrize("split,expected_length", [
        ("train", 20),  # 10 NORMAL + 10 PNEUMONIA
        ("val", 6),     # 3 NORMAL + 3 PNEUMONIA
        ("test", 4),    # 2 NORMAL + 2 PNEUMONIA
    ])
    def test_dataset_different_splits(self, mock_data_dir_with_splits, split, expected_length):
        """Test dataset with different splits (train, val, test)."""
        dataset = ChestXRayDataset(mock_data_dir_with_splits, split=split)
        assert len(dataset) == expected_length, \
            f"Dataset split '{split}' should have {expected_length} samples, got {len(dataset)}"
    
    def test_dataset_getitem_returns_correct_types(self, mock_data_dir):
        """Test __getitem__ returns tensor and label with correct types."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
        assert isinstance(label, int), "Label should be an integer"
    
    @pytest.mark.parametrize("expected_shape", [(3, 224, 224)])
    def test_dataset_getitem_image_shape(self, mock_data_dir, expected_shape):
        """Test that transformed images have correct shape (3, 224, 224)."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        image, _ = dataset[0]
        
        assert image.shape == expected_shape, \
            f"Image should have shape {expected_shape}, got {image.shape}"
    
    def test_dataset_getitem_image_normalized(self, mock_data_dir):
        """Test that images are normalized to expected range."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        image, _ = dataset[0]
        
        # Normalized images should have values roughly in range [-3, 3]
        assert image.min() >= -5.0, f"Image min value {image.min()} is outside expected range"
        assert image.max() <= 5.0, f"Image max value {image.max()} is outside expected range"
    
    def test_dataset_labels_encoding(self, mock_data_dir):
        """Test that labels are correctly encoded (NORMAL=0, PNEUMONIA=1)."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        
        unique_labels = set(dataset.labels)
        assert 0 in unique_labels, "Label 0 (NORMAL) should be present in dataset"
        assert 1 in unique_labels, "Label 1 (PNEUMONIA) should be present in dataset"
        assert len(unique_labels) == 2, f"Dataset should have exactly 2 unique labels, got {len(unique_labels)}"
    
    def test_dataset_labels_count(self, mock_data_dir):
        """Test that we have correct number of each label."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        
        normal_count = dataset.labels.count(0)
        pneumonia_count = dataset.labels.count(1)
        
        expected_count = 5
        assert normal_count == expected_count, \
            f"Should have {expected_count} NORMAL samples, got {normal_count}"
        assert pneumonia_count == expected_count, \
            f"Should have {expected_count} PNEUMONIA samples, got {pneumonia_count}"
    
    def test_dataset_samples_are_sorted(self, mock_data_dir):
        """Test that samples are sorted for reproducibility."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        
        sample_names = [s.name for s in dataset.samples]
        assert sample_names == sorted(sample_names), \
            "Sample paths should be sorted for reproducibility"
    
    def test_dataset_empty_directory(self):
        """Test dataset behavior with empty directory."""
        temp_dir = tempfile.mkdtemp()
        data_path = Path(temp_dir)
        
        # Create empty train directory
        (data_path / "train" / "NORMAL").mkdir(parents=True, exist_ok=True)
        (data_path / "train" / "PNEUMONIA").mkdir(parents=True, exist_ok=True)
        
        dataset = ChestXRayDataset(data_path, split="train")
        
        assert len(dataset) == 0, "Empty directory should result in dataset with 0 samples"
        assert len(dataset.samples) == 0, "Empty directory should have no sample paths"
        assert len(dataset.labels) == 0, "Empty directory should have no labels"
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_dataset_missing_class_directory(self):
        """Test dataset behavior when one class directory is missing."""
        temp_dir = tempfile.mkdtemp()
        data_path = Path(temp_dir)
        
        # Only create NORMAL directory, not PNEUMONIA
        normal_dir = data_path / "train" / "NORMAL"
        normal_dir.mkdir(parents=True, exist_ok=True)
        
        # Create some NORMAL images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode='L')
            img.save(normal_dir / f"normal_{i}.jpeg")
        
        dataset = ChestXRayDataset(data_path, split="train")
        
        assert len(dataset) == 3, "Dataset should only contain NORMAL samples"
        assert all(label == 0 for label in dataset.labels), \
            "All labels should be 0 (NORMAL) when PNEUMONIA directory is missing"
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_dataset_all_indices_accessible(self, mock_data_dir):
        """Test that all indices in dataset are accessible."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        
        for i in range(len(dataset)):
            image, label = dataset[i]
            assert image is not None, f"Image at index {i} should not be None"
            assert label in [0, 1], f"Label at index {i} should be 0 or 1, got {label}"
    
    def test_dataset_index_out_of_bounds(self, mock_data_dir):
        """Test that accessing out of bounds index raises IndexError."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]
    
    @pytest.mark.parametrize("index", [-1, -2])
    def test_dataset_negative_index(self, mock_data_dir, index):
        """Test that negative indexing works correctly."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        
        image, label = dataset[index]
        assert isinstance(image, torch.Tensor), f"Negative index {index} should return a tensor"
        assert isinstance(label, int), f"Negative index {index} should return an integer label"
    
    def test_dataset_transform_grayscale_to_rgb(self, mock_data_dir):
        """Test that grayscale images are converted to RGB (3 channels)."""
        dataset = ChestXRayDataset(mock_data_dir, split="train")
        image, _ = dataset[0]
        
        assert image.shape[0] == 3, \
            f"Images should have 3 channels (RGB), got {image.shape[0]} channels"
    
    def test_dataset_preprocess_creates_directories(self):
        """Test that preprocess method creates necessary directories."""
        temp_dir = tempfile.mkdtemp()
        data_path = Path(temp_dir)
        
        # Create mock raw data structure
        raw_path = data_path / "raw"
        chest_xray_path = raw_path / "chest_xray"
        
        # Create source directories with mock data
        for split in ["train", "val", "test"]:
            split_dir = chest_xray_path / split
            normal_dir = split_dir / "NORMAL"
            pneumonia_dir = split_dir / "PNEUMONIA"
            
            normal_dir.mkdir(parents=True, exist_ok=True)
            pneumonia_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a mock image
            img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode='L')
            img.save(normal_dir / "test.jpeg")
        
        # Create dataset and run preprocess
        output_folder = data_path / "processed"
        dataset = ChestXRayDataset(data_path / "processed" / "train")
        dataset.preprocess(output_folder)
        
        # Check that processed directories were created
        assert output_folder.exists(), "Output folder should be created"
        assert (output_folder / "train").exists(), "Train split should be created"
        assert (output_folder / "val").exists(), "Val split should be created"
        assert (output_folder / "test").exists(), "Test split should be created"
        
        # Check that data was copied
        assert (output_folder / "train" / "NORMAL" / "test.jpeg").exists(), \
            "Data should be copied to processed folder"
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_dataset_preprocess_skips_existing_directories(self):
        """Test that preprocess doesn't overwrite existing directories."""
        temp_dir = tempfile.mkdtemp()
        data_path = Path(temp_dir)
        
        # Create mock raw data
        raw_path = data_path / "raw"
        chest_xray_path = raw_path / "chest_xray"
        train_dir = chest_xray_path / "train" / "NORMAL"
        train_dir.mkdir(parents=True, exist_ok=True)
        
        img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode='L')
        img.save(train_dir / "test.jpeg")
        
        # Create output folder with existing data
        output_folder = data_path / "processed"
        existing_train = output_folder / "train" / "NORMAL"
        existing_train.mkdir(parents=True, exist_ok=True)
        
        marker_file = existing_train / "existing_marker.txt"
        marker_file.write_text("existing")
        
        # Run preprocess
        dataset = ChestXRayDataset(data_path / "processed" / "train")
        dataset.preprocess(output_folder)
        
        # Check that existing data was not overwritten
        assert marker_file.exists(), "Existing files should not be deleted"
        assert marker_file.read_text() == "existing", "Existing files should not be modified"
        
        # Cleanup
        shutil.rmtree(temp_dir)

