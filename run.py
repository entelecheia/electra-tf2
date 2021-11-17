import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

# from create_tfrecords import TFRecords

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# import dotenv
# dotenv.load_dotenv(override=True)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # Pretty print config using Rich library
	if cfg.get("print_config"):
		print('## hydra configuration ##')
		print(OmegaConf.to_yaml(cfg))

	if cfg.get("print_resolved_config"):
		print('## hydra configuration resolved ##')
		args = OmegaConf.to_container(cfg, resolve=True)
		pprint(args)
		print()

	print(f"Current working directory : {os.getcwd()}")
    # Train model
    # return train(config)
	# Init lightning model

	if cfg.get("datamodule"):
		print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
		dm = hydra.utils.instantiate(cfg.datamodule)
		# dm.run()


if __name__ == "__main__":
    main()