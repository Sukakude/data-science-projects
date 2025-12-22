# MLOps Pipeline with ZenML & MLflow
This project demonstrates an end-to-end MLOps workflow using ZenML for pipeline orchestration and MLflow for experiment tracking, model registry, and model serving.

It covers:
<ul>
  <li>Data ingestion and preprocessing</li>
  <li>Model training and evaluation</li>
  <li>Model deployment</li>
  <li>Model serving and inference</li>
</ul>

# Technologies Used
<ul>
  <li>Python 3.10+</li>
  <li>ZenML</li>
  <li>MLflow</li>
  <li>scikit-learn</li>
  <li>pandas / numpy</li>
</ul>

#### <i>Please Note: This project uses ZenML’s MLflow model deployer, which relies on background daemon processes.</i>
The project works best on <i>Linux</i> and <i>macOS</i>.

# Running the Training & Deployment Pipeline
<ol>
  <li><strong>Install the dependencies</strong>: pip install -r requirements.txt</li>
  <li>
    <strong>Initialize ZenML</strong>: 
    <ul>
      <li>zenml init</li>
      <li>zenml integration install mlflow -y</li>
    </ul>
  </li>
  <li><strong>Run training & deployment</strong>: python run_deployment.py --config deploy</li>
</ol>

These steps will train a Linear Regression model, evaluate the performance and deploy the model if it meets the criteria. Then, the model will be registered into MLFlow.

### To inspect experiments and models:
mlflow ui

# Future Improvements
<ul>
  <li>Dockerized inference service</li>
  <li>Kubernetes deployment</li>
  <li>CI/CD integration</li>
</ul>

# Author
SH3PO

