# Sales-Analysis
Data munching and analysis for sales in business.

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# run project 
python src/main.py
```

# Workflow idea
## Reading the Receipts (pandas)
- make dataframe, sort dates from oldest to newest if needed.
- group all the Laptop sales together and give me the total and all the North region sales and tell me the average profit

## Predict (sklearn)
- LabelEncoder to convert words into numbers (North = 0, South = 1, East = 2)
- RandomForestRegressor decision trees for past examples, which is needed to predict next month, it uses all those little trees and outputs a highly a guess of what the sales will be if predicted right.

## Charts (matplotlib + seaborn)

- matplotlib and seaborn for data visualization, saved in `reports` folder.

## To-Do

- [ ] Add seperate window for charts (Web app?)
- [ ] Make a better front end
- [ ] Make it flexible with various sales dataset
- [ ] A better if not same ML algorithm