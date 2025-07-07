check: 
    ruff check --fix *.py --exclude '*.ipynb' 
    black .

test:
    pytest tests/