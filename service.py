import bentoml
import numpy as np

import bentoml.models


@bentoml.service(
    name="iris_classifier",
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class IrisClassifier:
    iris_model = bentoml.models.BentoModel("iris_classifier:latest")
    service_name = "iris_classifier"

    def __init__(self) -> None:
        import joblib

        self.model = joblib.load(self.iris_model.path_of("saved_model.pkl"))

    @bentoml.api
    def classify(self, input_series: np.ndarray) -> np.ndarray:
        with bentoml.monitor(self.service_name) as mon:
            mon.log(
                self.service_name,
                name="service_name",
                role="service_name",
                data_type="text",
            )
            mon.log(
                input_series.tolist(),
                name="input_array",
                role="feature",
                data_type="numerical_sequence",
            )

            result = self.model.predict(input_series)

            mon.log(
                result.tolist(),
                name="output_array",
                role="prediction",
                data_type="numerical",
            )
            return result
