# Sales-Analysis
Data munching and analysis for sales in business.

# Workflow idea
## Reading the Receipts (pandas)
- make dataframe, sort dates from oldest to newest if needed.
- group all the Laptop sales together and give me the total and all the North region sales and tell me the average profit

## Predict (sklearn)
- LabelEncoder to convert words into numbers (North = 0, South = 1, East = 2)
- RandomForestRegressor decision trees for past examples, which is needed to predict next month, it uses all those little trees and outputs a highly a guess of what the sales will be if predicted right.

## Charts (matplotlib + seaborn)

- matplotlib and seaborn for data visualization, saves in `reports` folder.