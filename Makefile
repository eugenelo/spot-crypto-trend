setup: requirements.txt
	pip install -r third_party/requirements.txt

lint:
	black . --check

fix:
	black .
