import pandas as pd

# Loading the csv files
matches = pd.read_csv('fifa_matches.csv')
results = pd.read_csv('results.csv')
rankings = pd.read_csv('fifa_rankings_june_2026.csv')
# Checking column and row numbers
print(matches.shape)
print(results.shape)
print(rankings.shape)

# Creating team_a data from the matches dataset
team_a = matches[
    ["Match ID", "Date", "Team_A", "Team_B", "Team_A_Goals"]
].copy()

team_a = team_a.rename(columns={
    "Match ID": "Match_ID",
    "Team_A": "Team",
    "Team_B": "Opponent",
    "Team_A_Goals": "Goals_Scored"
})
# Creating team_b data from the matches dataset
team_b = matches[
    ["Match ID", "Date", "Team_A", "Team_B", "Team_B_Goals"]
].copy()

team_b = team_b.rename(columns={
    "Match ID": "Match_ID",
    "Team_A": "Opponent",
    "Team_B": "Team",
    "Team_B_Goals": "Goals_Scored"
})

team_b = team_b[
    ["Match_ID", "Date", "Team", "Opponent", "Goals_Scored"]
]

# Combining the team_a and team_b data
dataset = pd.concat([team_a, team_b], ignore_index=True)
dataset = dataset.sort_values(
    "Match_ID", kind="stable"
).reset_index(drop=True)

print(dataset.shape)
print(dataset.head())
print(dataset.tail())

# Converting the date to pandas datetime
dataset["Date"] = pd.to_datetime(
    dataset['Date'], format="%d/%m/%Y"
)
results["date"] = pd.to_datetime(
    results['date'], format="%Y-%m-%d"
)
# Checking and fixing team name consistency
missing_rankings = set(dataset["Team"]) - set(rankings["Team"])
print(missing_rankings)

rankings['Team'] = rankings["Team"].replace({
    "Korea Republic": "South Korea"
})

historical_teams = set(results["home_team"]) | set(results["away_team"])
print(set(dataset["Team"]) - historical_teams)

team_names = {
    "United States": "USA",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Iran": "IR Iran",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Turkey": "Türkiye"
}

results["home_team"] = results["home_team"].replace(team_names)
results["away_team"] = results["away_team"].replace(team_names)

historical_teams = set(results["home_team"]) | set(results["away_team"])
print(set(dataset["Team"]) - historical_teams)


# Consistent history

# Keep historical results from before the tournament
pre_tournament_results = results[
    results["date"] < dataset["Date"].min()
].copy()

# Use the matches file for tournament dates and scores
tournament_results = matches[
    ["Date", "Team_A", "Team_B", "Team_A_Goals", "Team_B_Goals"]
].copy()

tournament_results = tournament_results.rename(columns={
    "Date": "date",
    "Team_A": "home_team",
    "Team_B": "away_team",
    "Team_A_Goals": "home_score",
    "Team_B_Goals": "away_score"
})

tournament_results["date"] = pd.to_datetime(
    tournament_results["date"], format="%d/%m/%Y"
)

# Combine pre-tournament results with tournament results
history = pd.concat(
    [pre_tournament_results, tournament_results],
    ignore_index=True
)


# --------------------------------------------------
# Building the variables and adding them to the dataset
# --------------------------------------------------

# Adding the FIFA ranking variables
rank_lookup = rankings.set_index("Team")['FIFA_Rank']

dataset["Team_FIFA_Rank"] = dataset["Team"].map(rank_lookup)

dataset["Opponent_FIFA_Rank"] = dataset["Opponent"].map(
    rank_lookup
)
# Team average goals scored in the last 5 matches
def average_goals_last5(team, match_date):
    team_matches = history[
        (history['date'] < match_date) &
        (
            (history['home_team'] == team) |
            (history['away_team'] == team)
        )
    ]
    team_last5 = team_matches.sort_values("date").tail(5)
    team_goals = team_last5['home_score'].where(
        team_last5['home_team'] == team,
        team_last5['away_score']
    )
    team_avg = team_goals.mean()
    return team_avg

dataset['Team_Avg_Goals_Last5'] = dataset.apply(
    lambda row: average_goals_last5(row['Team'], row['Date']),
    axis=1
)
# Opponent average goals conceded in the last 5 matches
def average_conceded_last5(team, match_date):
    team_matches = history[
        (history['date'] < match_date) &
        (
            (history['home_team'] == team) |
            (history['away_team'] == team)
        )
    ]
    team_last5 = team_matches.sort_values("date").tail(5)
    team_conceded = team_last5['away_score'].where(
        team_last5['home_team'] == team,
        team_last5['home_score']
    )

    team_conceded_avg = team_conceded.mean()
    return team_conceded_avg


dataset['Opponent_Avg_Conceded_Last5'] = dataset.apply(
    lambda row: average_conceded_last5(
        row['Opponent'], row['Date']
    ),
    axis=1
)

# Team Scoring Consistency Last 5 Matches
def scoring_matches_last5(team, match_date):
    team_matches = history[
        (history['date'] < match_date) &
        (
            (history['home_team'] == team) |
            (history['away_team'] == team)
        )
    ]
    team_last5 = team_matches.sort_values("date").tail(5)

    team_goals = team_last5['home_score'].where(
        team_last5['home_team'] == team,
        team_last5['away_score']
    )
    return (team_goals > 0).sum()


dataset['Team_Scoring_Consistency_Last5'] = dataset.apply(
    lambda row: scoring_matches_last5(
        row['Team'], row['Date']
    ),
    axis=1
)


# Team Wins Last 5 Matches
def wins_last5(team, match_date):
    team_matches = history[
        (history['date'] < match_date) &
        (
            (history['home_team'] == team) |
            (history['away_team'] == team)
        )
    ]

    team_last5 = team_matches.sort_values("date").tail(5)
    team_goals = team_last5['home_score'].where(
        team_last5['home_team'] == team,
        team_last5['away_score']
    )
    team_conceded = team_last5['away_score'].where(
        team_last5['home_team'] == team,
        team_last5['home_score']
    )
    return (team_goals > team_conceded).sum()


dataset['Team_Wins_Last5'] = dataset.apply(
    lambda row: wins_last5(row['Team'], row['Date']),
    axis=1
)

# Opponent Losses Last 5 Matches
def losses_last5(team, match_date):
    team_matches = history[
        (history['date'] < match_date) &
        (
            (history['home_team'] == team) |
            (history['away_team'] == team)
        )
    ]
    team_last5 = team_matches.sort_values("date").tail(5)
    team_goals = team_last5['home_score'].where(
        team_last5['home_team'] == team,
        team_last5['away_score']
    )
    team_conceded = team_last5['away_score'].where(
        team_last5['home_team'] == team,
        team_last5['home_score']
    )
    return (team_goals < team_conceded).sum()


dataset['Opponent_Losses_Last5'] = dataset.apply(
    lambda row: losses_last5(row['Opponent'], row['Date']),
    axis=1
)

# Team Goal Differences Last 5 Matches
def goal_differences_last5(team, match_date):
    team_matches = history[
        (history['date'] < match_date) &
        (
            (history['home_team'] == team) |
            (history['away_team'] == team)
        )
    ]
    team_last5 = team_matches.sort_values("date").tail(5)
    team_goals = team_last5['home_score'].where(
        team_last5['home_team'] == team,
        team_last5['away_score']
    )
    team_conceded = team_last5['away_score'].where(
        team_last5['home_team'] == team,
        team_last5['home_score']
    )
    return (team_goals - team_conceded).sum()

dataset['Team_Goal_Differences_Last5'] = dataset.apply(
    lambda row: goal_differences_last5(
        row['Team'], row['Date']
    ),
    axis=1
)


# --------------------------------------------------
# CHECKING THE DATASET
# --------------------------------------------------
print(dataset.shape)
print(dataset.columns.tolist())
print(dataset.isna().sum())

# Checking that five previous matches are available
def history_count_last5(team, match_date):
    previous_matches = history[
        (history["date"] < match_date) &
        (
            (history["home_team"] == team) |
            (history["away_team"] == team)
        )
    ]
    return len(previous_matches.tail(5))


history_counts = dataset.apply(
    lambda row: history_count_last5(row["Team"], row["Date"]),
    axis=1
)
print(history_counts.value_counts())

# Checking Scotland's history before its Brazil match
print(history[
    (history["date"] < pd.Timestamp("2026-06-25")) &
    (
        (history["home_team"] == "Scotland") |
        (history["away_team"] == "Scotland")
    )
].sort_values("date").tail(5)[
    ["date", "home_team", "away_team", "home_score", "away_score"]
])

print(dataset[
    (dataset["Team"] == "Scotland") &
    (dataset["Opponent"] == "Brazil")
][[
    "Team_Avg_Goals_Last5",
    "Team_Scoring_Consistency_Last5",
    "Team_Wins_Last5",
    "Team_Goal_Differences_Last5"
]])

print(dataset.groupby("Match_ID").size().value_counts())

print(dataset.duplicated(subset=["Match_ID", "Team"]).sum())

# Saving the dataset to a CSV format so that it can be used to model the linear regression 2.2
dataset.to_csv("regression_2_2_dataset.csv", index=False)