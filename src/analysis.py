import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Setup

os.makedirs("results", exist_ok=True)

plt.style.use("ggplot")

# Load dataset

df = pd.read_csv("data/processed/asset_manager_earnings_dataset.csv")

print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

print("\nDATASET OVERVIEW\n")
print(df.head())

print("\nBasic statistics\n")
print(df.describe())

# CAR Summary Statistics

print("\nCAR Statistics\n")

car_stats = df[["CAR_3d", "CAR_5d"]].describe()

print(car_stats)

car_stats.to_csv("results/car_summary_statistics.csv")

# Plot CAR Distribution

plt.hist(df["CAR_3d"].dropna(), bins=15)

plt.title("Distribution of CAR (3 Day)")
plt.xlabel("CAR_3d")
plt.ylabel("Frequency")

plt.savefig("results/car_distribution.png", dpi=300)
plt.close()

# 1. Sentiment vs Stock Reaction

print("\nSentiment vs Stock Reaction\n")

df["sentiment_bucket"] = pd.cut(
    df["management_sentiment_score"],
    bins=[-1, -0.2, 0.2, 1],
    labels=["Negative", "Neutral", "Positive"]
)

sentiment_analysis = df.groupby("sentiment_bucket")["earnings_reaction"].mean()

print(sentiment_analysis)

sentiment_analysis.to_csv("results/sentiment_vs_returns.csv")

# Plot

plt.scatter(
    df["management_sentiment_score"],
    df["earnings_reaction"]
)

plt.xlabel("Management Sentiment Score")
plt.ylabel("Stock Return")
plt.title("Earnings Sentiment vs Stock Reaction")

plt.savefig("results/sentiment_vs_reaction.png", dpi=300)
plt.close()

# Sentiment vs CAR

print("\nSentiment vs CAR\n")

sentiment_car = df.groupby("sentiment_bucket")["CAR_3d"].mean()

print(sentiment_car)

sentiment_car.to_csv("results/sentiment_vs_car.csv")

plt.scatter(
    df["management_sentiment_score"],
    df["CAR_3d"]
)

plt.xlabel("Management Sentiment Score")
plt.ylabel("CAR (3 Day)")
plt.title("Sentiment vs Abnormal Returns")

plt.savefig("results/sentiment_vs_car.png", dpi=300)
plt.close()

# 2. Net Flows vs Stock Reaction

print("\nFlows vs Stock Reaction\n")

flows_corr = df[["net_flows_billion_usd", "earnings_reaction"]].corr()
print(flows_corr)

flows_corr.to_csv("results/flows_correlation.csv")

# Positive vs Negative flows

df["flow_direction"] = df["net_flows_billion_usd"] > 0

flow_analysis = df.groupby("flow_direction")["earnings_reaction"].mean()

print("\nPositive vs Negative Flows Impact")
print(flow_analysis)

flow_analysis.to_csv("results/flow_direction_vs_returns.csv")

# 3. AUM Growth vs Stock Reaction

print("\nAUM Growth vs Stock Reaction\n")

aum_corr = df[["AUM_growth_pct", "earnings_reaction"]].corr()

print(aum_corr)

aum_corr.to_csv("results/aum_growth_correlation.csv")

# 4. Company-Level Analysis

print("\nCompany Average Earnings Reaction\n")

company_returns = df.groupby("company")["earnings_reaction"].mean().sort_values()

print(company_returns)

company_returns.to_csv("results/company_average_returns.csv")

# Company-level CAR

print("\nCompany Average CAR\n")

company_car = df.groupby("company")["CAR_3d"].mean().sort_values()

print(company_car)

company_car.to_csv("results/company_average_car.csv")

company_car.plot(kind="bar")

plt.title("Average CAR by Firm")
plt.ylabel("CAR (3 Day)")

plt.savefig("results/company_car.png", dpi=300)
plt.close()

# 5. Sentiment vs Flows

print("\nSentiment vs Flows Correlation\n")

sentiment_flow_corr = df[["management_sentiment_score", "net_flows_billion_usd"]].corr()

print(sentiment_flow_corr)

sentiment_flow_corr.to_csv("results/sentiment_flow_correlation.csv")

print("\nAnalysis Complete\n")

# 6. Flow Surprise Analysis

print("\nFlow Surprise Analysis\n")

df["avg_company_flows"] = df.groupby("company")["net_flows_billion_usd"].transform("mean")

# Flow surprise vs historical average

df["flow_surprise"] = df["net_flows_billion_usd"] - df["avg_company_flows"]

# Buckets

df["flow_surprise_bucket"] = pd.cut(
    df["flow_surprise"],
    bins=[-100, -1, 1, 100],
    labels=["Negative Surprise", "Neutral", "Positive Surprise"]
)

flow_surprise_analysis = df.groupby("flow_surprise_bucket")["earnings_reaction"].mean()

print(flow_surprise_analysis)

flow_surprise_analysis.to_csv("results/flow_surprise_vs_returns.csv")

# Flow surprise vs CAR

flow_surprise_car = df.groupby("flow_surprise_bucket")["CAR_3d"].mean()

print("\nFlow Surprise vs CAR")
print(flow_surprise_car)

flow_surprise_car.to_csv("results/flow_surprise_vs_car.csv")

# Plot

flow_surprise_analysis.plot(kind="bar")

plt.title("Stock Reaction vs Flow Surprise")
plt.ylabel("Average Return")

plt.savefig("results/flow_surprise_chart.png", dpi=300)
plt.close()

# 7. Machine Learning Models

print("\nMachine Learning Model\n")

df["abs_sentiment"] = df["management_sentiment_score"].abs()

features = [
    "management_sentiment_score",
    "abs_sentiment",
    "net_flows_billion_usd",
    "AUM_growth_pct"
]

# Drop missing values

ml_df = df.dropna(subset=features + ["CAR_3d"])

X = ml_df[features]
y = ml_df["CAR_3d"]

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Linear Regression

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

lin_predictions = lin_model.predict(X_test)

lin_r2 = r2_score(y_test, lin_predictions)

print("\nLinear Regression R2 Score:")
print(lin_r2)

print("\nFeature Importance (Linear Regression):")

for feature, coef in zip(features, lin_model.coef_):
    print(f"{feature}: {coef}")

# Random Forest

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest R2 Score:")
print(rf_r2)

# Gradient Boosting

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)

gb_r2 = r2_score(y_test, gb_predictions)

print("\nGradient Boosting R2 Score:")
print(gb_r2)

# Feature importance plot

importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values()

importance.plot(kind="barh")

plt.title("Random Forest Feature Importance")

plt.savefig("results/feature_importance.png", dpi=300)
plt.close()

# Save ML results

model_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "R2": [
        lin_r2,
        rf_r2,
        gb_r2
    ]
})

model_results.to_csv("results/model_performance.csv", index=False)

# Save enriched dataset

df.to_csv("results/enriched_earnings_dataset.csv", index=False)

print("\nAll results saved to /results folder\n")