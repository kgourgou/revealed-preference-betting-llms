check: 
    ruff check --fix *.py --exclude '*.ipynb' 
    black .

test:
    pytest tests/

clean: 
    trash __pycache__/
    trash src/__pycache__/
    trash tests/__pycache__/
