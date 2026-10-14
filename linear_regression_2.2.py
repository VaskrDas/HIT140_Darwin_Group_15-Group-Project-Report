import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# ------------LOADING AND SPLITTING DATA-------------
# --------------------------------------------------

# Loading the dataset and restoring the date column
dataset = pd.read_csv(
    "regression_2_2_dataset.csv",
    parse_dates=["Date"]
)

# Checking if everything is loaded successfully
print(dataset.shape)
dataset.info()
print(dataset.isna().sum())

# Selecting the eight explanatory variables
explanatory_variables = [
    "Team_FIFA_Rank",
    "Opponent_FIFA_Rank",
    "Team_Avg_Goals_Last5",
    "Opponent_Avg_Conceded_Last5",
    "Team_Scoring_Consistency_Last5",
    "Team_Wins_Last5",
    "Opponent_Losses_Last5",
    "Team_Goal_Differences_Last5"
]

# Reserving 20% of matches for testing
# Both team rows from a match stay in the same set
match_ids = dataset["Match_ID"].unique()

train_ids, test_ids = train_test_split(
    match_ids,
    test_size=0.2,
    random_state=0
)

train_data = dataset[
    dataset["Match_ID"].isin(train_ids)
].copy()

test_data = dataset[
    dataset["Match_ID"].isin(test_ids)
].copy()

# Checking for 83 training matches and 21 testing matches
print("Training matches:", len(train_ids))
print("Testing matches:", len(test_ids))

# Separating the explanatory variables and actual goals
X_train = train_data[explanatory_variables]
y_train = train_data["Goals_Scored"]

X_test = test_data[explanatory_variables]
y_test = test_data["Goals_Scored"]

# Checking for 166 training rows and 42 testing rows
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# ------------EXPLORATORY DATA ANALYSIS--------------
# --------------------------------------------------

# Checking how often each number of goals appears
goal_counts = y_train.value_counts().sort_index()
print(goal_counts)

# Plotting the goals scored in the training data
goal_counts.plot(kind="bar")

plt.xlabel("Goals Scored")
plt.ylabel("Number of team-match rows")
plt.title("Goals Scored in the Training Data")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Checking the summary of the explanatory variables
print(X_train.describe().T)

# Checking the distribution of each explanatory variable
X_train.hist(bins=10, figsize=(14, 10))

plt.tight_layout()
plt.show()

# Checking correlations with goals and between variables
# Strong correlations between variables can indicate collinearity
correlations = train_data[
    explanatory_variables + ["Goals_Scored"]
].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlations,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0
)

plt.title("Correlations in the Training Data")
plt.tight_layout()
plt.show()

# Comparing straight and curved relationships with FIFA ranks
# These plots explore each rank separately
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, variable in zip(
    axes, ["Team_FIFA_Rank", "Opponent_FIFA_Rank"]
):
    sns.regplot(
        data=train_data,
        x=variable,
        y="Goals_Scored",
        ci=None,
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red", "label": "Straight line"},
        ax=ax
    )

    sns.regplot(
        data=train_data,
        x=variable,
        y="Goals_Scored",
        order=2,
        ci=None,
        scatter=False,
        line_kws={"color": "green", "label": "Curved line"},
        ax=ax
    )

    ax.set_title(variable)
    ax.legend()

plt.tight_layout()
plt.show()


# --------------------------------------------------
# ------------LINEAR REGRESSION MODEL----------------
# --------------------------------------------------

# Building the original linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Displaying the intercept and coefficients
print("Intercept:", model.intercept_)

coefficients = pd.Series(
    model.coef_,
    index=explanatory_variables
)

print(coefficients)

# Predicting training goals to calculate residuals
y_train_pred = model.predict(X_train)
residuals = y_train - y_train_pred

# Checking residuals against predicted goals
plt.scatter(y_train_pred, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Predicted goals")
plt.ylabel("Residual: actual - predicted")
plt.title("Training Residuals vs Predicted Goals")
plt.tight_layout()
plt.show()

# Checking the distribution of residuals
plt.hist(residuals, bins=15, edgecolor="black")
plt.axvline(0, color="red", linestyle="--")

plt.xlabel("Residual: actual - predicted")
plt.ylabel("Number of team-match rows")
plt.title("Training Residual Distribution")
plt.tight_layout()
plt.show()

# Checking coefficients, p-values and adjusted R-squared
# Grouping the standard errors because each match has two rows
X_train_constant = sm.add_constant(X_train)

ols_model = sm.OLS(
    y_train, X_train_constant
).fit(
    cov_type="cluster",
    cov_kwds={"groups": train_data["Match_ID"]}
)

print(ols_model.summary())


# --------------------------------------------------
# ------------CROSS-VALIDATION EXPERIMENTS-----------
# --------------------------------------------------

# Using five rounds while keeping match pairs together
cv = GroupKFold(n_splits=5)
groups = train_data["Match_ID"]

cv_results = []

for fold, (train_positions, validation_positions) in enumerate(
    cv.split(X_train, y_train, groups),
    start=1
):
    # Selecting the learning and validation rows for this round
    X_fold_train = X_train.iloc[train_positions]
    y_fold_train = y_train.iloc[train_positions]

    X_fold_validation = X_train.iloc[validation_positions]
    y_fold_validation = y_train.iloc[validation_positions]

    # Building a fresh linear regression model
    fold_model = LinearRegression()
    fold_model.fit(X_fold_train, y_fold_train)

    linear_predictions = fold_model.predict(X_fold_validation)

    # Trying a log transformation of the team's FIFA rank
    X_fold_train_log = X_fold_train.copy()
    X_fold_validation_log = X_fold_validation.copy()

    X_fold_train_log["Team_FIFA_Rank"] = np.log(
        X_fold_train_log["Team_FIFA_Rank"]
    )

    X_fold_validation_log["Team_FIFA_Rank"] = np.log(
        X_fold_validation_log["Team_FIFA_Rank"]
    )

    log_model = LinearRegression()
    log_model.fit(X_fold_train_log, y_fold_train)

    log_predictions = log_model.predict(X_fold_validation_log)

    # Trying a power transformation of the explanatory variables
    # Learning the transformation from this round's learning rows
    transformer = PowerTransformer()

    X_fold_train_power = transformer.fit_transform(X_fold_train)
    X_fold_validation_power = transformer.transform(
        X_fold_validation
    )

    power_model = LinearRegression()
    power_model.fit(X_fold_train_power, y_fold_train)

    power_predictions = power_model.predict(
        X_fold_validation_power
    )

    # Using the learning-set mean as the baseline prediction
    baseline_predictions = [
        y_fold_train.mean()
    ] * len(y_fold_validation)

    # Collecting predictions from the regression experiments
    fold_predictions = [
        ("Linear regression", linear_predictions),
        ("Log team rank", log_predictions),
        ("Power transformation", power_predictions),
        ("Mean baseline", baseline_predictions)
    ]

    # Comparing decision trees with three different depths
    for depth in [2, 3, 4]:
        tree_model = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=10,
            random_state=0
        )

        tree_model.fit(X_fold_train, y_fold_train)

        tree_predictions = tree_model.predict(X_fold_validation)

        fold_predictions.append(
            (f"Decision tree depth {depth}", tree_predictions)
        )

    # Calculating the same measurements for every model
    for model_name, predictions in fold_predictions:
        mse = mean_squared_error(
            y_fold_validation, predictions
        )

        cv_results.append({
            "Fold": fold,
            "Model": model_name,
            "MAE": mean_absolute_error(
                y_fold_validation, predictions
            ),
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "R_squared": r2_score(
                y_fold_validation, predictions
            )
        })

# Converting the results into a dataframe
cv_results = pd.DataFrame(cv_results)

metrics_columns = ["MAE", "MSE", "RMSE", "R_squared"]

# Comparing the average results across five rounds
cv_summary = cv_results.groupby("Model")[metrics_columns].mean()

print("\nAverage cross-validation results:")
print(cv_summary.sort_values("MAE").round(3).to_string())

# Checking how much performance varies across rounds
cv_variation = cv_results.groupby("Model")[
    ["MAE", "RMSE", "R_squared"]
].std()

print("\nStandard deviation across rounds:")
print(cv_variation.round(3).to_string())


# --------------------------------------------------
# ------------FINAL TEST EVALUATION------------------
# --------------------------------------------------

# Using MAE as the main measurement for model selection
# Power transformation had the lowest average validation MAE
# Depth 2 had the lowest average validation MAE among the trees
# Keeping these choices fixed for the final test evaluation

# Predicting test goals using the original linear regression
linear_predictions = model.predict(X_test)

# Learning the power transformation from all training rows
final_transformer = PowerTransformer()

X_train_power = final_transformer.fit_transform(X_train)
X_test_power = final_transformer.transform(X_test)

final_power = LinearRegression()
final_power.fit(X_train_power, y_train)

power_predictions = final_power.predict(X_test_power)

# Building the selected decision tree
final_tree = DecisionTreeRegressor(
    max_depth=2,
    min_samples_leaf=10,
    random_state=0
)

final_tree.fit(X_train, y_train)
tree_predictions = final_tree.predict(X_test)

# Predicting the training mean for every test row
baseline_predictions = [y_train.mean()] * len(y_test)

# Evaluating all four models on the same test rows
test_results = []

for model_name, predictions in [
    ("Linear regression", linear_predictions),
    ("Power transformation", power_predictions),
    ("Decision tree depth 2", tree_predictions),
    ("Mean baseline", baseline_predictions)
]:
    mse = mean_squared_error(y_test, predictions)

    test_results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "R_squared": r2_score(y_test, predictions)
    })

test_results = pd.DataFrame(test_results)

print("\nFinal test results:")
print(test_results.round(3).to_string(index=False))


# --------------------------------------------------
# ------------CHECKING TEST PREDICTIONS--------------
# --------------------------------------------------

# Comparing actual and predicted goals for the selected model
plt.figure(figsize=(7, 6))
plt.scatter(y_test, power_predictions, alpha=0.7)

# Drawing a reference line where actual and predicted goals match
lower = min(y_test.min(), power_predictions.min())
upper = max(y_test.max(), power_predictions.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    color="red",
    linestyle="--"
)

plt.xlabel("Actual goals")
plt.ylabel("Predicted goals")
plt.title("Power-Transformed Linear Regression: Test Predictions")
plt.tight_layout()
plt.show()