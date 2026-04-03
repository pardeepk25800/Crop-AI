# tests/test_models.py — Model Unit Tests for CropAI
# Run: python -m pytest tests/test_models.py -v

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDiseaseModelArchitecture:
    """Tests for the disease detection CNN architecture."""

    def test_model_creation_efficientnet(self):
        """CropDiseaseModel should create with EfficientNet-B3 backbone."""
        from disease_model import CropDiseaseModel
        model = CropDiseaseModel(num_classes=38, backbone="efficientnet_b3", pretrained=False)
        assert model is not None
        assert model.backbone_name == "efficientnet_b3"

    def test_model_creation_resnet(self):
        """CropDiseaseModel should create with ResNet-50 backbone."""
        from disease_model import CropDiseaseModel
        model = CropDiseaseModel(num_classes=38, backbone="resnet50", pretrained=False)
        assert model is not None
        assert model.backbone_name == "resnet50"

    def test_model_creation_mobilenet(self):
        """CropDiseaseModel should create with MobileNet-V3 backbone."""
        from disease_model import CropDiseaseModel
        model = CropDiseaseModel(num_classes=38, backbone="mobilenet_v3_large", pretrained=False)
        assert model is not None

    def test_model_invalid_backbone(self):
        """CropDiseaseModel should raise ValueError for unknown backbone."""
        from disease_model import CropDiseaseModel
        with pytest.raises(ValueError, match="Unknown backbone"):
            CropDiseaseModel(num_classes=38, backbone="invalid_net")

    def test_model_forward_pass(self):
        """Forward pass should produce correct output shape."""
        import torch
        from disease_model import CropDiseaseModel

        model = CropDiseaseModel(num_classes=38, backbone="efficientnet_b3", pretrained=False)
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, 38), f"Expected (2, 38), got {output.shape}"

    def test_model_freeze_unfreeze(self):
        """Freeze/unfreeze backbone should change requires_grad status."""
        from disease_model import CropDiseaseModel

        model = CropDiseaseModel(num_classes=38, backbone="efficientnet_b3", pretrained=False)

        model.freeze_backbone()
        frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        assert frozen > 0, "Backbone should have frozen parameters"

        model.unfreeze_backbone(layers_from_end=2)
        unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        assert unfrozen > 0, "Some backbone layers should be unfrozen"


class TestYieldModelArchitecture:
    """Tests for the yield prediction ensemble."""

    def test_ensemble_creation(self):
        """YieldEnsemble should initialize correctly."""
        from yield_model import YieldEnsemble
        ensemble = YieldEnsemble()
        assert ensemble is not None
        assert not ensemble._trained

    def test_ensemble_predict_before_fit(self):
        """Predicting before fitting should raise AssertionError."""
        from yield_model import YieldEnsemble
        ensemble = YieldEnsemble()
        with pytest.raises(AssertionError, match="Call fit"):
            ensemble.predict(np.array([[1, 2, 3, 4, 5]]))


class TestDataTransforms:
    """Tests for image transforms."""

    def test_train_transforms(self):
        """Training transforms should produce correct tensor shape."""
        import torch
        from disease_model import get_transforms

        transform = get_transforms(train=True)
        from PIL import Image
        img = Image.new("RGB", (300, 300), color=(34, 139, 34))
        tensor = transform(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_val_transforms(self):
        """Validation transforms should produce correct tensor shape."""
        import torch
        from disease_model import get_transforms

        transform = get_transforms(train=False)
        from PIL import Image
        img = Image.new("RGB", (300, 300), color=(34, 139, 34))
        tensor = transform(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)


class TestConfigConstants:
    """Tests for configuration constants."""

    def test_disease_classes_count(self):
        """Should have exactly 38 disease classes."""
        from config import DISEASE_CLASSES, NUM_CLASSES
        assert len(DISEASE_CLASSES) == 38
        assert NUM_CLASSES == 38

    def test_image_size(self):
        """Image size should be 224 for EfficientNet."""
        from config import IMAGE_SIZE
        assert IMAGE_SIZE == 224

    def test_paths_exist(self):
        """Data and model directories should be created."""
        from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR
        assert os.path.isdir(DATA_DIR)
        assert os.path.isdir(MODEL_DIR)
        assert os.path.isdir(RESULTS_DIR)
        assert os.path.isdir(LOGS_DIR)
