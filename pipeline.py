import azureml.core
from azureml.core import Workspace, Datastore, Experiment
from azureml.pipeline.core import Pipeline
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.train.estimator import Estimator
from azureml.train.dnn import PyTorch
from azureml.widgets import RunDetails
from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule
from azureml.pipeline.steps import EstimatorStep

#Get workspace configuration
ws = Workspace.from_config() 
experiment = Experiment(ws, 'species-classifier') 

#setup compute target //get the compute targets which are already created
data_fetch_compute = ws.compute_targets['cpu_cluster']
training_compute = ws.compute_targets["gpu_cluster"]

#Get the datastore where data sits
bird_data_store = Datastore(ws, "birdimages")

#Add a dataReference for the datastore where data lives
bird_data = DataReference(
    datastore=Datastore(ws, bird_data_store),
    data_reference_name="bird_data",
    path_on_datastore="./bird_data")

#Pipeline data is the intermediate output of an step.
resized_data = PipelineData(
    "resized_data",
    datastore=bird_data_store,
    output_name="resized_data")

#Configure the steps in pipeline
#data fetching step
data_fetch_step = PythonScriptStep(
    script_name="image_fetcher.py",
    arguments=["--input", bird_data],
    inputs=[bird_data],
    compute_target=data_fetch_compute,
    source_directory="./scripts",
    allow_reuse=True
)

#data resize step
image_resize_step = PythonScriptStep(
    script_name="image_resize.py",
    arguments=["--input", bird_data, "--output", resized_data],
    inputs=[bird_data],
    outputs=[resized_data],
    compute_target=data_fetch_compute,
    source_directory="./scripts",
    allow_reuse=True
)

#model training using estimator step

#pyTorch estimator
pyTorchEstimator = PyTorch(source_directory='./scripts', 
     script_params='./params/est_params.yml', 
     compute_target= training_compute, 
     entry_script='./scripts/bird_classifier_train.py', 
     framework_version='1.2', 
     use_docker=True, 
     use_gpu=True)

estimator_step = EstimatorStep(name="Estimator_Train", 
                         estimator=pyTorchEstimator, 
                         estimator_entry_script_arguments=["--datadir", resized_data],
                         runconfig_pipeline_params=None, 
                         inputs=[resized_data], 
                         compute_target=training_compute)

#pipeline with many steps
modelRetrainSteps = [data_fetch_step, image_resize_step,estimator_step]

# Build the pipeline
retrain_pipeline = Pipeline(workspace=ws, steps=[modelRetrainSteps])

#Run pipeline
pipeline_run = experiment.submit(retrain_pipeline)
pipeline_run.wait_for_completion()

#Get run details
RunDetails(pipeline_run).show()

#Publish pipeline
published_retrain_pipeline = pipeline_run.publish_pipeline(
     name="model_retrain_pipeline",
     description="Retraining the image classification model",
     version="1.0")


#Model retraining
recurrence = ScheduleRecurrence(frequency="Month", interval=1, week_days=["Wednesday"], time_of_day="23:30") # Runs every other day at 10:30pm

schedule = Schedule.create(workspace=ws, name="Retrain_Schedule",
                           pipeline_id=published_retrain_pipeline, 
                           experiment_name='species-classifier',
                           recurrence=recurrence,
                           wait_for_provisioning=True,
                           description="Schedule retraining of the model")

# You may want to make sure that the schedule is provisioned properly
print("Created schedule with id: {}".format(schedule.id))