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
import numpy as np
import time
from base_runner import *
import wml.wml_utils as wmlu
from datadef import get_class_name, set_class_name



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
        lpid,
        gpu_mem,
        *args,
        **kwargs,
):
    wmlu.wait_gpu_mem_free(gpu,gpu_mem,delay=60)
    if lpid>1:
        wmlu.wait_pid_exit(lpid)
    np.random.seed(int(time.time()))
    methods = {key: item for (key, item) in methods}

    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed, test)

    device = utils.set_torch_device(gpu)

    result_collect = []
    data = {'Class': [], 'Distribution': [], 'Foreground': []}
    df = pd.DataFrame(data)
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        try:
            set_class_name(dataloaders["training"].dataset.classname)
            utils.fix_seeds(seed, device)
            dataset_name = dataloaders["training"].name
            imagesize = dataloaders["training"].dataset.imagesize
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
                flag = 0., 0., 0., 0., 0., -1.
                if GLASS.backbone.seed is not None:
                    #utils.fix_seeds(GLASS.backbone.seed, device)
                    pass
    
                GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name,run_save_path=run_save_path)
                flag = GLASS.trainer(dataloaders["training"], dataloaders["testing"], dataloaders["base_training"],dataset_name)
        except Exception as e:
            wmlu.print_error(f"{e}")


    # save distribution judgment xlsx after all categories
    if len(df['Class']) != 0:
        os.makedirs('./datasets/excel', exist_ok=True)
        xlsx_path = './datasets/excel/' + dataset_name.split('_')[0] + '_distribution.xlsx'
        df.to_excel(xlsx_path, index=False)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
