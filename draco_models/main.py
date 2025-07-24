import draco_models.job
import argparse
import sys
import apscheduler.schedulers.background
import apscheduler.triggers.cron
from draco_models.config import load_config, InputConfig


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
    job.export_models(models, job.config.save_models)


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
    config = load_config(args.config)
    # Validate the configuration
    input_config = InputConfig(**config)
    # Run the main function with the validated configuration
    if len(input_config.cron_schedule) != 0:
        # Get a scheduler instance and a Cron trigger from the configuration
        scheduler = apscheduler.schedulers.background.BlockingScheduler()
        trigger = apscheduler.triggers.cron.CronTrigger.from_crontab(
            input_config.cron_schedule
        )
        # Schedule the main function to run periodically based on the cron schedule limited to one instance
        scheduler.add_job(
            main, trigger, args=[input_config], id="draco_trainer", max_instances=1
        )
        scheduler.start()
    else:
        # If no cron schedule is provided, run the main function
        main(input_config)


if __name__ == "__main__":
    cli_entrypoint()
