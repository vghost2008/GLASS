from datetime import datetime
import pandas as pd
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"
import logging
import sys
import click
import torch
import warnings
import backbones
import glass
import utils
from base_runner import *



@main.result_callback()
def run(
        methods,
        results_path,
        gpu,
        seed,
        log_group,
        log_project,
        run_name,
        test,
        ckpt_path,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed, test)

    device = utils.set_torch_device(gpu)

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        utils.fix_seeds(seed, device)
        dataset_name = dataloaders["predict"].name
        imagesize = dataloaders["predict"].dataset.imagesize
        glass_list = methods["get_glass"](imagesize, device)

        LOGGER.info(
            "Selecting dataset [{}] ({}/{}) {}".format(
                dataset_name,
                dataloader_count + 1,
                len(list_of_dataloaders),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        )

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, GLASS in enumerate(glass_list):
            GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name,run_save_path=run_save_path,tb_dir="tb_pred")
            try:
                GLASS.run_predict(dataloaders["predict"], dataloaders["predict"].dataset.classname)
            except Exception as e:
                print(f"ERROR: Run {dataset_name} faild, info {e}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
