# Agentic AutoML Assistant

A beginner-friendly Streamlit app that lets you upload a CSV dataset, run an end-to-end AutoML pipeline, and get model comparisons, saved artifacts, and a Markdown report. It also includes an optional LLM-powered assistant for dataset and model Q&A.

## Features
- CSV upload and data preview
- Automatic problem-type detection (classification vs regression)
- Preprocessing with missing value handling and one-hot encoding
- Train/test split and baseline model training
- Model evaluation, best-model selection, and model saving
- Visualizations for missing values, target distribution, and model comparison
- Auto-generated Markdown report with download button
- Optional LLM assistant chat (uses environment variables for API keys)

## Tech Stack
- Python
- Streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- openai (optional, for the assistant)

## Folder Structure
```
agentic-automl-assistant/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── agent.py
│   ├── assistant.py
│   ├── data_analyzer.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── model_trainer.py
│   ├── preprocessor.py
│   └── report_generator.py
├── models/
├── reports/
└── sample_data/
```

## Installation Steps
1. Create a virtual environment:
	```bash
	python -m venv .venv
	```
2. Activate it:
	```bash
	.venv\Scripts\activate
	```
3. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

Optional: enable the assistant by setting an API key in your environment.
```bash
setx OPENAI_API_KEY "your_api_key_here"
```
You can also set a model name (default is gpt-4o-mini):
```bash
setx OPENAI_MODEL "gpt-4o-mini"
```

## How to Run the App
```bash
streamlit run app.py
```

## How to Use the App
1. Upload a CSV file.
2. Select a target column from the dropdown.
3. Review the dataset summary, charts, and detected problem type.
4. Let the AutoML pipeline run and view model comparisons.
5. Download the generated report and review the saved model path.
6. (Optional) Ask questions in the assistant chat panel.

## Example Workflow
1. Upload a housing prices CSV.
2. Choose the price column as the target.
3. The app detects regression and preprocesses the data.
4. Models train and compare performance.
5. The best model is saved in the models folder.
6. A report is generated in the reports folder and can be downloaded.

## Future Improvements
- Hyperparameter tuning and cross-validation
- Feature importance and explainability
- Support for time-series and text datasets
- Experiment tracking and run history
- Model deployment options

## Resume Project Description
Built a Streamlit-based AutoML assistant that automates dataset analysis, preprocessing, model training, evaluation, and reporting. The app provides visual insights, saves the best model artifact, and offers an optional LLM chat assistant for dataset and model Q&A.
