# Rooftop plant-forward cafe, Manish Nagar, Nagpur

Student: Harshita Chopda

A data-driven validation of a 70% vegan, 30% regular vegetarian rooftop cafe with no meat, eggs or alcohol. The survey data is mock data with intentional noise and outliers; upload real responses in the dashboard sidebar and all models rerun.

## Run locally (Python 3.10+)

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # Mac/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

`python run_all.py` regenerates the mock survey and prints the main results.

## Deploy a live link

1. Create a public GitHub repository and upload everything in this folder.
2. Go to share.streamlit.io and sign in with GitHub.
3. Choose Create app, then select the repository, branch `main` and main file `app.py`.
4. Deploy and share the URL.

## Files

- `survey_schema.py`: the 30 questions
- `generate_mock_survey.py`: mock responses with noise and outliers
- `analytics.py`: cleaning, classification, clustering, association rules, regression, sustainability model
- `app.py`: live Streamlit dashboard
- `data`, `survey`, `report`: data, questionnaire, report and screenshots

Business assumptions are in `A` at the top of `analytics.py`. They are planning estimates to be replaced with real quotes.
