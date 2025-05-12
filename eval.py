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
import wml.wml_utils as wmlu
from base_runner import *
from datadef import get_class_name,set_class_name

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
        *args,
        **kwargs,
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
    all_pauroc = []
    all_f1 = []
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        #utils.fix_seeds(seed, device)
        set_class_name(dataloaders["training"].dataset.classname)
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
            #wtt.register_forward_hook(GLASS,wtt.isfinite_hook)
            #wtt.register_forward_hook(GLASS,wtt.islarge_hook)
            #wtt.register_tensor_hook(GLASS,wtt.tensor_isfinite_hook)
            #######################################
            flag = 0., 0., 0., 0., 0., -1.
            if GLASS.backbone.seed is not None:
                #utils.fix_seeds(GLASS.backbone.seed, device)
                pass

            GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name,run_save_path=run_save_path,tb_dir="tb_eval")
            try:
                best_precision, best_recall,best_f1, p_auroc, pixel_ap, pixel_pro, epoch =  GLASS.tester(dataloaders["testing"], dataset_name,ckpt_path=ckpt_path)
            except:
                continue
            cur_score = GLASS.get_score(pauroc=p_auroc,f1=best_f1)
            print(f"dataset_name: {dataset_name}, M:{(best_f1+p_auroc)/2:.3f}, pixel_auroc: {p_auroc}, Precision: {best_precision}, Recall: {best_recall}, F1: {best_f1}, best_epoch: {epoch}\n" )
            all_pauroc.append(p_auroc)
            all_f1.append(best_f1)

    print(f"dataset_name: ALL, AUROC: {all_pauroc}, mean: {np.mean(all_pauroc):.4f}, F1: {all_f1}, mean: {np.mean(all_f1):.4f}")
    print(f"PAUROC")
    wmlu.show_list(all_pauroc,format="{:.3f}")
    print(f"F1")
    wmlu.show_list(all_f1,format="{:.3f}")
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
