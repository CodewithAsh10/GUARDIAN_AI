import sys
sys.path.insert(0, '.')
import importlib

pages = [
    'dashboard.pages.overview',
    'dashboard.pages.crime_trends',
    'dashboard.pages.state_analysis',
    'dashboard.pages.district_analysis',
    'dashboard.pages.hotspot_intelligence',
    'dashboard.pages.geographical_intelligence',
    'dashboard.pages.data_explorer',
]

for name in pages:
    try:
        mod = importlib.import_module(name)
        render = getattr(mod, 'render', None)
        print(f'{name}: OK (render={callable(render)})')
    except Exception as e:
        print(f'{name}: ERROR - {type(e).__name__}: {str(e)[:80]}')