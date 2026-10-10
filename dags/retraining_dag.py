from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="glucorisk_training_retraining",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * 1",
    catchup=False,
    default_args={
        "owner": "glucorisk",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["ml", "training", "mlflow"],
) as dag:
    train_and_register = BashOperator(
        task_id="train_and_register_production_model",
        bash_command="cd /opt/airflow && python src/train.py",
        env={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MLFLOW_MODEL_NAME": "GlucoRiskAI",
            "PYTHONPATH": "/opt/airflow",
        },
    )