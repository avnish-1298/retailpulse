install:
	pip install -r requirements.txt
	pip install -e .
extra:
	pip install -r requirements-extra.txt
pipeline:
	python -m retailpulse.pipeline
dashboard:
	streamlit run app/dashboard.py
mlflow-ui:
	mlflow ui --backend-store-uri file:./mlruns
test:
	pytest -q
docker:
	docker build -t retailpulse:latest .
