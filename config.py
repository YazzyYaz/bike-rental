import yaml

with open("config.yml") as f:
    config = yaml.load(f)
    dataset = config.get("dataset")