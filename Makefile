setup: requirements.txt
	pip install -r requirements.txt

lint:
	black . --check

fix:
	black .
