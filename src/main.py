import sys
from config import get_config
from train import Trainer
from training_wrapper import run_training_wrapper
from evaluation_wrapper import run_evaluation_wrapper
from logger import PerfomanceLogger


def main(configurations):
	Trainer.set_seed(configurations['random_seed'])
	PerfomanceLogger.configure_logger(configurations)
	perf_logger = PerfomanceLogger()
	if configurations['evaluation']:
		run_evaluation_wrapper(configurations, perf_logger)
	else:
		run_training_wrapper(configurations, perf_logger)


if __name__ == "__main__":
	config = get_config(sys.argv[1:])
	main(config)
