.PHONY: setup download validate features train evaluate backtest api dashboard test docker-up

setup:
	pip install -r requirements.txt
	cd dashboard && npm install

download:
	python -m src.data.collectors.historical

validate:
	python -m src.data.validators.validate

features:
	python -m src.features.pipeline

train:
	python -m src.models.trainer

evaluate:
	@echo "Evaluation is handled during the train step and saved to reports/"

backtest:
	python -m src.backtesting.engine

api:
	uvicorn src.api.main:app --reload

dashboard:
	cd dashboard && npm run dev

worker:
	python -m src.prediction.worker

test:
	PYTHONPATH=. pytest tests/

docker-up:
	docker-compose up --build -d
