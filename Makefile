dev-ui:
	PYTHONPATH=. streamlit run ui/app.py
dev-cli:
	PYTHONPATH=. python cli/main.py tests/fixtures/qa_tests.yaml --use-case qa --model examples/model.py
dev:
	make dev-ui & make dev-cli
