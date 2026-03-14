# Earnings Transcript Analysis for Asset Managers

## Overview

This project analyses earnings call transcripts from publicly listed asset managers to examine whether management sentiment and key business fundamentals help explain short-term stock price reactions.

Using an LLM-powered extraction pipeline, the project converts unstructured earnings transcripts into structured financial data including:

- management sentiment
- net fund flows
- AUM growth
- earnings metrics

These extracted variables are then linked to stock price reactions around earnings announcements. An event-study framework is used to compute both raw earnings reactions and abnormal returns (CAR) around the announcement window.
An event-study framework is used to compute both raw earnings reactions and abnormal returns (CAR) around the announcement window.


The goal is to explore whether information contained in earnings call transcripts can help explain or predict market reactions.

## Project Pipeline

The project follows the following workflow:

1. Raw earnings transcripts are stored in `data/raw/`. Need to be saved with e.g. '_q4_2025' in transcript name.
2. An OpenAI model extracts structured financial data from the transcripts
3. The extracted information is saved as JSON files and compiled into a dataset
4. Stock price reactions data is retrieved using `yfinance`
5. An event study is performed to compute:
   - raw earnings reactions
   - cumulative abnormal returns (CAR) over multiple event windows
6. The final dataset is analysed using statistical techniques and machine learning models

## Example Extracted Data

Example JSON output produced by the extraction pipeline:

example_json:
{
  "company": "BlackRock",
  "earnings_date": "2024-01-12",
  "management_sentiment_score": 0.42,
  "net_flows_billion_usd": 18.2,
  "AUM_growth_pct": 12.5,
  "earnings_reaction": 0.021,
  "CAR_3d":0.018,
  "CAR_5d":0.024,
}

## Key Research Questions

This project explores several research questions, including but not limited to:
- Does management sentiment in earnings calls influence short-term stock reactions?
- Are net fund flows associated with positive or negative market reactions?
- Does AUM growth correlate with earnings announcement returns?
- Can ML models predict stock reactions using extracted transcript data?
- Do earnings announcements generate abnormal returns around the event window?

## Analysis Performed
The analysis includes:
- Sentiment vs stock reaction analysis
- Sentiment vs abnormal return (CAR) analysis
- Net flows vs stock return correlations
- AUM growth vs stock reaction analysis
- Flow surprise analysis relative to historical averages
- Company-level abnormal return analysis
- Event-study summary statistics for CAR
- Example ML models to predict earnings reactions

The following models are implemented:
- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression
Model performance is evaluated using out-of-sample R² scores.

## Project Structure

project/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── src/
│ ├── main.py
│ ├── analysis.py
│ └── prompt_template.py
│
├── data/
│ ├── raw/
│ └── processed/
│
├── outputs/
│
└── results/

## Installation

Clone the repository:
    git clone https://github.com/yourusername/earnings-transcript-analysis.git
    cd earnings-transcript-analysis

Install dependencies:
    pip install -r requirements.txt

Create an environment file:
    cp .env.example .env

Then add your OpenAI API key to the `.env` file.

Example:
    OPENAI_API_KEY=your_api_key_here

## Usage

1. Run Transcript Extraction

This script extracts structured data from earnings transcripts and builds the dataset.
    python src/main.py

Outputs:

- JSON files containing extracted transcript data
- A combined dataset saved to `data/processed/`

### 2. Run Data Analysis

This script performs statistical analysis and machine learning modeling on the dataset.
    python src/analysis.py

Outputs:

- statistical summaries
- correlation results
- charts and visualizations
- machine learning model performance metrics

All analysis outputs are saved in the `results/` folder.

## Technologies Used

- Python  
- OpenAI API  
- Pandas  
- NumPy  
- yfinance  
- Matplotlib  
- Scikit-learn  

## Future Improvements

Some example otential extensions to the project include:

- Expanding the dataset to more asset managers and additional earnings periods  
- Comparing LLM sentiment extraction with financial NLP models such as FinBERT  
- Using embeddings for deeper transcript analysis  
- Adding additional machine learning models and feature engineering  

## Disclaimer

This project is for research and educational purposes only and should not be considered investment advice.