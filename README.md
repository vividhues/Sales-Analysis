# Sales-Analysis
Data munching and analysis for sales in business.

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# run project 
python src/main.py

# or if you want to run the web gui
streamlit run gui.py
```

# Workflow idea
## Reading the Receipts (pandas)
- make dataframe, sort dates from oldest to newest if needed.
- group all the Laptop sales together and give me the total and all the North region sales and tell me the average profit

## Predict (sklearn)
- LabelEncoder to convert words into numbers (North = 0, South = 1, East = 2)
- RandomForestRegressor decision trees for past examples, which is needed to predict next month, it uses all those little trees and outputs a highly a guess of what the sales will be if predicted right.

## Charts (matplotlib + seaborn) or (Streamlit + Plotly)

- matplotlib and seaborn for data visualization, saved in `reports` folder.
- if using streamlit, plotly renders the charts directly onto the web page.

## To-Do

- [x] Add seperate window for charts (Web app?)
- [x] Make a better front end
- [ ] Make it flexible with various sales dataset
- [ ] A better if not same ML algorithm