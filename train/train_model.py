import mlflow
import mlflow.statsmodels
import pickle
import pandas as pd

# Load the SARIMAX model
model_path = "model/sarimax_model.pkl"
with open(model_path, "rb") as file:
    sarimax_model = pickle.load(file)

# Create a valid input example for a time series model
input_example = pd.DataFrame({"start": [1], "end": [5]})  # SARIMAX needs 'start' and 'end'

# Start MLflow experiment
mlflow.set_experiment("SARIMAX_Model_Tracking")

with mlflow.start_run():
    mlflow.log_param("model_type", "SARIMAX")

    # Log the SARIMAX model using mlflow.statsmodels
    mlflow.statsmodels.log_model(
        artifact_path="SARIMAX_Model",
        statsmodels_model=sarimax_model,
        signature=None,  # Optional: You can use infer_signature()
        input_example=input_example
    )

    print("✅ SARIMAX Model logged successfully with MLflow!")
