import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from fastapi import FastAPI, Request

app = FastAPI()

train_feature_dataset = pd.read_csv("train_feature_dataset.csv")


def is_drift(origin_data: pd.DataFrame, current_data: pd.DataFrame) -> bool:
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=origin_data, current_data=current_data)
    data_drift_result = data_drift_report.as_dict()
    dataset_drift = data_drift_result["metrics"][0]
    return dataset_drift["result"]["dataset_drift"]


@app.post("/validation")
async def validate_data(request: Request):
    data_list: list[dict] = await request.json()
    features = [item for data in data_list for item in data["payload"]["input_array"]]
    feature_df = pd.DataFrame(features, columns=train_feature_dataset.columns)
    if is_drift(train_feature_dataset, feature_df):
        print("Dataset drift detected")
        # Slack 알림
    else:
        print("Dataset drift not detected")
