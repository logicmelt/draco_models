import draco_models.job
import argparse
import sys
from draco_models.config import load_config_with_default, InputConfig


def main(input_config: InputConfig) -> None:
    """
    Train machine learning models based on the provided configuration and export them to ONNX format.

    Args:
        input_config (InputConfig): Configuration for the training pipeline, including model parameters and database connection details.
    """
    # Create a Job instance
    job = draco_models.job.Job(input_config)

    # Train the models
    models = job.train()

    # Export the models to ONNX format
    job.export_models(models, job.run_dir)


def cli_entrypoint():
    """
    Command line entry point for the Draco Trainer.
    Parses command line arguments and runs the main training function.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train and export machine learning models."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file in YAML format.",
    )
    args, unknown = parser.parse_known_args()
    # Remove --config from the list of arguments
    sys.argv = [sys.argv[0]] + unknown
    # Load the configuration from a YAML file with defaults
    config = load_config_with_default(args.config)

    # Validate the configuration
    input_config = InputConfig(**config)
    # Run the main function with the validated configuration
    main(input_config)


if __name__ == "__main__":
    cli_entrypoint()
