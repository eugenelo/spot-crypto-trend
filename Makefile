setup: requirements.txt
	pip install -r third_party/requirements.txt

lint:
	black . --check --preview
	flake8 .

fix:
	isort .
	black . --preview
