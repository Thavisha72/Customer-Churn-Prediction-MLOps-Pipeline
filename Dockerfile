FROM apache/airflow:2.7.0

USER airflow

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    mlflow \
    dagshub \
    joblib \
    matplotlib \
    seaborn

WORKDIR /opt/airflow