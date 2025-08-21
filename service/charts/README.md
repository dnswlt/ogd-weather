# charts

This directory contains the main Python backend that reads
weather data from an sqlite or Postgres DB and renders
Vega/Vega-Lite charts.

## Dev notes

To quickly list "public" symbols in a module (e.g. to add
them to the `__all__` export list in `__init__.py`):

```bash
python -c "import ast, sys; p = ast.parse(open(sys.argv[1]).read()); names = set(); [names.add(n.name) for n in p.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]; [names.add(t.id) for n in p.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)]; print('\n'.join(sorted([f'    \"{n}\",' for n in names if not n.startswith('_')])))" service/charts/models/models.py
```
