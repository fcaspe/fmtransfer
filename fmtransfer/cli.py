"""
command line interface entrypoint.
"""
import logging
import os

from dotenv import load_dotenv
from pytorch_lightning.cli import LightningCLI

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.data_file", "model.init_args.data_file")
        parser.link_arguments("data.init_args.data_dir", "model.init_args.data_dir")
        parser.link_arguments(
            "data.init_args.envelope_type", "model.init_args.envelope_type"
        )


def run_cli():
    """ """
    _ = MyLightningCLI(save_config_overwrite=True)
    return


def main():
    """ """
    load_dotenv()
    run_cli()
