import os
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify, Response
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Initialize Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return "Flask API is running! Use /predict?dataset=indvsnz"

# Function to process dataset dynamically
def process_dataset(dataset_name, captain_name=None, vice_captain_name=None):
    dataset_path = f"./dataset/{dataset_name}.csv"
    
    if not os.path.exists(dataset_path):
        return {"error": f"Dataset {dataset_name} not found!"}

    # Load dataset
    df = pd.read_csv(dataset_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()  # Remove spaces from column names
    df_original = df.copy()  # Backup original data

    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df.fillna(df.median(), inplace=True)

    # Define features and target variable
    X = df.drop(columns=['Fantasy_score'])
    y = df['Fantasy_score']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Selection
    base_model = ExtraTreesRegressor(n_estimators=300, max_depth=10, random_state=42)
    selector = RFE(base_model, n_features_to_select=8, step=1)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Train Model
    model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=6)
    model.fit(X_train_selected, y_train)

    # Make Predictions
    df["Predicted_Fantasy_Score"] = model.predict(selector.transform(X))

    # Convert encoded values back to original labels
    df["Player Name"] = label_encoders["Player Name"].inverse_transform(df["Player Name"])
    df["Role"] = label_encoders["Role"].inverse_transform(df["Role"])
    df["Team"] = label_encoders["Team"].inverse_transform(df["Team"])

    # Prepare for optimization
    df = df[['Player Name', 'Role', 'Team', 'Predicted_Fantasy_Score']]
    df['Credits'] = np.random.randint(8, 12, size=len(df))

    # LP Optimization
    players = df['Player Name'].tolist()
    selection = {p: LpVariable(str(p), cat='Binary') for p in players}

    model = LpProblem("Team_Selection", LpMaximize)
    model += lpSum(df.loc[df['Player Name'] == p, 'Predicted_Fantasy_Score'].iloc[0] * selection[p] for p in players)

    # Role constraints
    model += lpSum(selection[p] for p in df[df['Role'] == 'Batsman']['Player Name']) == 5  
    model += lpSum(selection[p] for p in df[df['Role'] == 'Bowler']['Player Name']) == 4
    model += lpSum(selection[p] for p in df[df['Role'] == 'All-rounder']['Player Name']) == 4  
    model += lpSum(selection[p] for p in df[df['Role'] == 'Wicketkeeper']['Player Name']) == 2  

    model.solve()

    selected_players = [p for p in players if selection[p].varValue == 1]

    # Filter selected players
    selected_team_df = df[df['Player Name'].isin(selected_players)][['Player Name', 'Role', 'Team', 'Predicted_Fantasy_Score']]
    selected_team_df = selected_team_df.sort_values(by='Predicted_Fantasy_Score', ascending=False)

    # Ensure Captain & Vice-Captain are batsmen
    batsmen_df = selected_team_df[selected_team_df['Role'] == 'Batsman'].sort_values(by='Predicted_Fantasy_Score', ascending=False)

    if len(batsmen_df) >= 2:
        captain = batsmen_df.iloc[4]['Player Name']  # Highest-scoring batsman
        vice_captain = batsmen_df.iloc[2]['Player Name']  # Second highest

        selected_team_df.loc[selected_team_df['Player Name'] == captain, 'Role'] += " (C)"
        selected_team_df.loc[selected_team_df['Player Name'] == vice_captain, 'Role'] += " (VC)"
    else:
        print("Not enough batsmen to assign Captain & Vice-Captain. Adjust constraints!")

    return selected_team_df.to_dict(orient='records')

# Flask API Endpoint
@app.route('/predict', methods=['GET'])
def predict_team():
    dataset_name = request.args.get('dataset')
    captain_name = request.args.get('captain')
    vice_captain_name = request.args.get('vice_captain')

    if not dataset_name:
        return jsonify({"error": "Dataset name is required!"})

    result = process_dataset(dataset_name, captain_name, vice_captain_name)

    if "error" in result:
        return jsonify(result)

    # Convert DataFrame to CSV format
    csv_data = pd.DataFrame(result).to_csv(index=False)

    # Return as CSV response
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=selected_team.csv"}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
