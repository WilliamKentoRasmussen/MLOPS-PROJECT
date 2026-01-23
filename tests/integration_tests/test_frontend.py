import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_frontend_imports():
    """Ensure frontend can be imported without errors"""
    try:
        from src.main_project.frontend import (
            CLASS_NAMES,
            classify_image,
            get_backend_url,
            main,
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_constants():
    """Test important constants"""
    from src.main_project.frontend import CLASS_NAMES
    assert CLASS_NAMES == ["NORMAL", "PNEUMONIA"]

@patch('src.main_project.frontend.requests.post')
def test_backend_connection(mock_post):
    """Test that frontend can communicate with backend"""
    from src.main_project.frontend import classify_image

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'predicted_class': 'NORMAL',
        'confidence': 0.9
    }
    mock_post.return_value = mock_response

    result = classify_image(b'test', 'http://localhost:8000')
    assert result is not None
