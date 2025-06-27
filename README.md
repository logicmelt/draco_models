# Draco model trainer

## Dependencies setup

Pyenv is recommended to manage the Python version.
Poetry is required to install dependencies and launch the code.

```bash
pyenv shell 3.12.8
poetry install --no-root
```

## Training

With the package installed, it can be run by API, CLI-based commands and environmental variables.

```python
from draco_models.config import InputConfig, load_config_with_default
from draco_models.job import Job

yaml_path = "additional_files/training_config.yaml"
config = load_config_with_default(yaml_path)
# Validate the configuration
input_config = InputConfig(**config)
# # Create a Job instance
job = Job(input_config)
# Train the models
models = job.train()
# Export the models to ONNX format
job.export_models(models, input_config.save_dir)
```
Via CLI we have two options:
- Use a configuration file 
```bash
draco_trainer --config additional_files/training_config.yaml
```
- Use a configuration file and override from command line or environmental variables.
```bash
export INFLUXDB___TOKEN=8086
draco_trainer --config additional_files/training_config.yaml --save_dir "test_dir"
```
Within the save directory onnx models, json file with scores, log file and config json will be generated.