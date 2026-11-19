# Titanic passenger data

This Streamlit app explores the included `titanic.csv` in four tabs: Overview,
Explore, Preprocessing, and Data Explorer. Shared sidebar filters update the
passenger metrics, charts, and data table. The Data Explorer provides Raw,
Cleaned, and Transformed views. The app fills missing age and embarked values,
adds Min-Max age/fare features and age groups, and previews dummy-coded sex and
embarked values. Findings describe observed associations, not causation.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

Prerequisite: install `uv` if you don't already have it.

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Sync the dependencies

   ```
   $ uv sync
   ```

2. Run the app

   ```
   $ uv run streamlit run app.py
   ```
