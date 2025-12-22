import numpy as np
import pandas as pd
import json
from zenml import pipeline, step # type:ignore
from zenml.config import DockerSettings # type:ignore

from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT # type:ignore
from zenml.integrations.constants import MLFLOW # type:ignore
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer # type:ignore

from zenml.integrations.mlflow.services import MLFlowDeploymentService # type:ignore
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step # type:ignore

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

from .utils import get_data_for_test

# CONFIGURE THE DOCKER SETTINGS
docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"])
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction
    
@step
def deployment_trigger(accuracy: float):
    min_accuracy: float=0
    """This function implements a trigger that checks the accuracy of the model and decides whether to deploy or not."""
    return accuracy >= min_accuracy


# SETUP CONTINOUS DEPLOYMENT PIPELINE
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(min_accuracy: float=0.92, workers: int=1, timeout: int=DEFAULT_SERVICE_START_STOP_TIMEOUT):
    path = r'.\data\olist_customers_dataset.csv'
    df = ingest_df(path=path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    r2_score, rsme, mse = evaluate_model(model, X_test, y_test)
    
    # CRITERIA FOR DEPLOYING THE MODEL: r2_score > min_accuracy
    deployment_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model, 
        deploy_decision=deployment_decision,
        workers = workers,
        timeout = timeout
    )


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str
):
    batch_data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=service, data=batch_data)
    return prediction