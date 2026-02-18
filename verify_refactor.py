
import sys
import unittest
from unittest.mock import MagicMock

# Mock streamlit before imports
sys.modules['streamlit'] = MagicMock()
sys.modules['streamlit.query_params'] = {}

# Mock other specific molecules if needed
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()

import os
# Ensure we are in the project root
sys.path.append(os.getcwd())

class TestRefactor(unittest.TestCase):
    def test_model_registry(self):
        print("Testing model_registry...")
        from models.model_registry import MODEL_INFO, get_model_info
        self.assertIn('bigram', MODEL_INFO)
        self.assertIn('mlp', MODEL_INFO)
        print("✅ model_registry OK")

    def test_app_imports(self):
        print("Testing app.py imports...")
        import app
        self.assertTrue(hasattr(app, 'main'))
        print("✅ app.py imports OK")
        
    def test_viz_imports(self):
        print("Testing viz modules imports...")
        from models import bigram_viz
        from models import mlp_viz
        self.assertTrue(hasattr(bigram_viz, 'render_bigram'))
        self.assertTrue(hasattr(mlp_viz, 'render_mlp'))
        print("✅ viz modules imports OK")

if __name__ == '__main__':
    unittest.main()
