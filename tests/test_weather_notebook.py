"""Synthetic and static checks; no real weather or country CSV is bundled."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/knowledge-based-weather-recommendation-system.ipynb'
CELLS = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']

def source(i):
    return ''.join(CELLS[i]['source'])

class WeatherTests(unittest.TestCase):
    def test_cells_compile(self):
        for i, cell in enumerate(CELLS):
            if cell['cell_type'] == 'code':
                with self.subTest(cell=i):
                    compile(source(i), f'cell-{i}', 'exec')

    def test_geographic_splits_are_unique_and_disjoint(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'WEATHER_DATA_DIR': directory}):
            ns = {}
            exec(compile(source(1), 'geography-cell', 'exec'), ns)
            splits = [set(map(tuple, ns[key].to_numpy())) for key in ('train_df', 'valid_df', 'test_df')]
            self.assertFalse(splits[0] & splits[1])
            self.assertFalse(splits[0] & splits[2])
            self.assertFalse(splits[1] & splits[2])
            all_real = set.union(*splits)
            self.assertIn(('Washington, D.C.', 'isCapitalOf', 'United States'), all_real)
            self.assertIn(('Canberra', 'isCapitalOf', 'Australia'), all_real)
            self.assertIn(('Abuja', 'isCapitalOf', 'Nigeria'), all_real)
            self.assertNotIn(('New York', 'isCapitalOf', 'United States'), all_real)
            self.assertNotIn(('Sydney', 'isCapitalOf', 'Australia'), all_real)
            self.assertNotIn(('Lagos', 'isCapitalOf', 'Nigeria'), all_real)
            self.assertFalse(set(map(tuple, ns['synthetic_df'].to_numpy())) & all_real)

if __name__ == '__main__':
    unittest.main()
