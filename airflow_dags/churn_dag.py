from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="customer_churn_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    run_dvc_pipeline = BashOperator(
        task_id="run_dvc_pipeline",
        bash_command="cd /opt/project && dvc repro",
    )

    run_dvc_pipeline