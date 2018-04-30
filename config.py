import yaml
import coloredlogs, logging

with open("config.yml") as f:
    config = yaml.load(f)
    dataset = config.get("dataset")
    hyperparameters = config.get("hyperparameters")
    dataset_link = config.get("dataset_link")

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')
coloredlogs.install(level='DEBUG')