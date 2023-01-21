.PHONY: setup clean test check

test:
	python -m pytest -ra -s

setup: requirements.txt
	pip install -r requirements.txt

clean:
	rm -rf __pycache__

format:
	black .

check:test format
