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
import wml.wtorch.train_toolkit as wtt
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
        #utils.fix_seeds(seed, device)
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
            #######################################
            wtt.register_forward_hook(GLASS,wtt.isfinite_hook)
            wtt.register_forward_hook(GLASS,wtt.islarge_hook)
            wtt.register_tensor_hook(GLASS,wtt.tensor_isfinite_hook)
            #######################################
            flag = 0., 0., 0., 0., 0., -1.
            if GLASS.backbone.seed is not None:
                #utils.fix_seeds(GLASS.backbone.seed, device)
                pass

            GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name,run_save_path=run_save_path,tb_dir="tb_eval")
            i_auroc, i_ap, p_auroc, p_ap, p_pro, epoch = GLASS.tester(dataloaders["testing"], dataset_name,ckpt_path=ckpt_path)
            print(f"dataset_name: {dataset_name}, M:{(i_auroc+p_auroc)/2:.3f}, image_auroc: {i_auroc}, image_ap: {i_ap}, pixel_auroc: {p_auroc}, pixel_ap: {p_ap}, pixel_pro: {p_pro}, best_epoch: {epoch}\n" )
            sys.stdout.flush()
            result_collect.append(
                    {
                        "dataset_name": dataset_name,
                        "image_auroc": i_auroc,
                        "image_ap": i_ap,
                        "pixel_auroc": p_auroc,
                        "pixel_ap": p_ap,
                        "pixel_pro": p_pro,
                        "best_epoch": epoch,
                    }
                )

            if epoch > -1:
                for key, item in result_collect[-1].items():
                    if isinstance(item, str):
                        continue
                    elif isinstance(item, int):
                        print(f"{key}:{item}")
                    else:
                        print(f"{key}:{round(item * 100, 2)} ", end="")

            # save results csv after each category
            print("\n")
            result_metric_names = list(result_collect[-1].keys())[1:]
            result_dataset_names = [results["dataset_name"] for results in result_collect]
            result_scores = [list(results.values())[1:] for results in result_collect]
            utils.compute_and_store_final_results(
                run_save_path,
                result_scores,
                result_metric_names,
                row_names=result_dataset_names,
            )

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
