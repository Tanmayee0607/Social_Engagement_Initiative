# Data-Driven Social Engagement Initiative

## Project purpose
This prototype analyzes content performance and audience comments, ranks posts using a simple project-defined score, compares content formats, and shows data-based recommendations in a Streamlit dashboard.

## Important data note
The included CSV files are **synthetic demonstration data** created only so the project runs without an externally supplied dataset. They are not real Instagram/YouTube data and must not be presented as real observations. Replace them with a permitted public dataset or collected data if your faculty requires real data.

## Requirements
- Python 3.10 or newer recommended
- Internet access only to install packages (the app itself uses local CSV files)

## Run on Windows
1. Extract the ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal > New Terminal.
4. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

5. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

7. Live dashboard link , https://socialengagementinitiative-hsqhk4aobupvgytddmw45w.streamlit.app/. Open it in your browser.

## Run the NLP script separately
From the project root:

```bash
python scripts/sentiment_model.py
```

## Project modules
1. Content performance tracker: the CSV is the local structured dataset; the app reads it as the prototype's data ingestion layer.
2. Virality prediction/ranking: a transparent weighted score prioritizes shares and saves. It is a ranking heuristic, not a trained viral-probability model.
3. Audience sentiment analyzer: TF-IDF + Logistic Regression predicts `Relatable` or `Neutral` from labelled example comments.
4. A/B testing framework: the dashboard compares formats descriptively. Historical comparisons are not a controlled A/B test.
5. Engagement recommender: surfaces the highest-average-engagement topic and format in the selected data.
6. Growth dashboard: shows topic metrics and engagement trends over time.
7. Trend forecasting: the current prototype visualizes engagement over time and includes hashtags, but does not yet provide a reliable external hashtag forecast. This is a stretch feature; add genuine time-series data if required.

## Metric definitions
- Engagement rate (%) = (likes + comments + shares + saves) / reach * 100
- Project-defined viral score = (3*shares + 2*saves + likes + comments) / reach * 100
- Retention rate is stored as a fraction between 0 and 1 in the sample data.

## Limitations
- Synthetic data and synthetic comment labels are for demonstrating the pipeline only.
- The comment classifier is a baseline and may not generalize to real comments.
- Correlation and historical differences do not prove causation.
- APIs and scraping are not connected in this deadline-friendly prototype.
