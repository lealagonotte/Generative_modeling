import json, os, sys
import argparse
import logging

import pickle as pkl
import pandas as pd
import numpy as np

from pathlib import Path

from tqdm.auto import tqdm
from torch import from_numpy
from torch import device as torch_device
from torch.cuda import is_available as cuda_is_available
from torch.optim import Adam

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from generate_dataset.generation_dataset_script import generate_data, generate_Nx2D_data
from generate_dataset.utils import inpainting_corruption, inpainting_corruption_pointwise, inpainting_corruption_Nx2D, inpainting_corruption_pointwise_Nx2D
from training.viz import viz_sample_2D, viz_sample_Nx2D, viz_loss_curves
from training.ambient_diffusion import NoiseScheduler, FurtherCorrupter, Sampler, AmbientLoss
from training.module import Denoiser, FlatDenoiserNx2D, PointNetDenoiserNx2D
from training.training import train, load_dataset
from training.metrics import wasserstein_distance, sliced_wasserstein_distance, chamfer_distance
from training.utils import TqdmToLogger

METRIC_DICT = {"wd": wasserstein_distance, "swd": sliced_wasserstein_distance, "cd": chamfer_distance}

global SEEDS
SEEDS = []

global LOGGER
LOGGER = None

def compute_metrics(dataset_path, metric_list, sampler, n_samples, n_steps, 
                    module, further_corrupter, noise_scheduler):
    with open(dataset_path, "rb") as f:
        ref_data = pkl.load(f)
    
    mode = ref_data.get("mode", "2D")
    device = next(module.parameters()).device
    
    if mode == "Nx2D":
        n_points_per_cloud = ref_data.get("n_points_per_cloud", 200)
        shape = (n_samples, n_points_per_cloud, 2)
    else:
        shape = (n_samples, 2)

    # Sample mask (same logic as training.py sample())
    A = further_corrupter.init_operator(shape, device)
    A_sample = further_corrupter.get_operator(A)

    # Run sampling with trajectory
    module.eval()
    LOGGER.info("Sampling..")
    samples = sampler.sample(
        shape, n_steps, A_sample, module, noise_scheduler
    )

    with open(dataset_path, "rb") as f:
        ref_data = pkl.load(f)
    X_ref = from_numpy(ref_data["X"])

    metric_results = {}
    for metric in metric_list:
        LOGGER.info(f"Computing {metric}...")
        metric_results[metric] = METRIC_DICT[metric](samples, X_ref)

    return metric_results
    
def launch_experiments(dataset_path, p, delta_list, 
                       metric_list, ranking_metric, 
                       batch_size, 
                       schedule, schedule_kwargs,
                       module_kwargs, device,
                       adam_kwargs,
                       epochs, patience,
                       sampler, n_samples, n_steps,
                       ):
    # Load data
    train_loader, val_loader, dataset_type = load_dataset(dataset_path, batch_size)

    # Setup components
    noise_scheduler = NoiseScheduler(schedule, **schedule_kwargs)
    
    sampler = Sampler(sampler)

    results = []
    loss_curves = []
    best_so_far = {"metrics":{m:np.inf for m in metric_list}}
    worst_so_far = {"metrics":{m:-np.inf for m in metric_list}}

    tqdm_out = TqdmToLogger(LOGGER,level=logging.INFO)
    for method in tqdm(["ambient", "naive"], file=tqdm_out, desc="Method"):
        LOGGER.info(f"Running {method} method".upper())
        for delta in tqdm(delta_list, file=tqdm_out, desc="Delta"):
            LOGGER.info(f"Using {delta} further corruption..")
            further_corrupter = FurtherCorrupter(dataset_type, p=delta)
        
            training_kwargs = module_kwargs.copy()
            model_type = training_kwargs.pop("model", "mlp")
            data_dim = training_kwargs.pop("data_dim", 2)
            
            if model_type == "mlp":
                module = Denoiser(data_dim=data_dim, **training_kwargs).to(device)
            elif model_type == "flat_nx2d":
                n_points = training_kwargs.pop("n_points_per_cloud", 200)
                module = FlatDenoiserNx2D(n_points=n_points, data_dim=data_dim, **training_kwargs).to(device)
            elif model_type == "pointnet_nx2d":
                module = PointNetDenoiserNx2D(data_dim=data_dim, **training_kwargs).to(device)
            else:
                raise ValueError(f"Unknown model type {model_type}")
            
            ambient_loss = AmbientLoss(further_corrupter.apply_operator_func)
            optimizer = Adam(module.parameters(), **adam_kwargs)

            # Train
            module, train_losses, val_losses = train(
                train_loader, val_loader, epochs, patience,
                ambient_loss, optimizer, module, noise_scheduler, further_corrupter,
                method, logger=LOGGER
            )

            metric_dict = compute_metrics(dataset_path, metric_list, 
                                          sampler, n_samples, n_steps, 
                                          module, further_corrupter, noise_scheduler)

            if metric_dict[ranking_metric] < best_so_far["metrics"][ranking_metric]:
                best_so_far = {"dataset_path": dataset_path, 
                               "method":method, "p":p, "delta": delta, 
                               "module":module, "metrics":metric_dict, "device":device,
                               "corrupter":further_corrupter, "scheduler": noise_scheduler}
                
                
            if metric_dict[ranking_metric] > worst_so_far["metrics"][ranking_metric]:
                worst_so_far = {"dataset_path": dataset_path, 
                               "method":method, "p":p, "delta": delta, 
                               "module":module, "metrics":metric_dict, "device":device,
                               "corrupter":further_corrupter, "scheduler": noise_scheduler}

            row_dict = {"method":method, "p": p, "delta": delta}
            for metric, metric_value in metric_dict.items():
                row_dict[metric] = metric_value
            results.append(row_dict)

            loss_curves.append({"method":method, "p": p, "delta": delta, 
                                "train":train_losses, "val":val_losses})
            
    return results, best_so_far, worst_so_far, loss_curves

def make_table(results, metric_list, folder):
    filename = folder / "results_table.csv"

    df = pd.DataFrame(results)

    agg_dict = {}
    for metric in metric_list:
        agg_dict[f"Avg {metric}"] = (metric, 'mean')
        agg_dict[f"Std {metric}"] = (metric, 'std')

    aggregated_df = df.groupby(["method", "p", "delta"]).agg(**agg_dict)
    
    aggregated_df.to_csv(str(filename))
    LOGGER.info(f"Table saved at {str(filename)}")

def plot_loss_curves(loss_curves, folder):
    viz_loss_curves(loss_curves, folder)

def visualize_best_worse(best, worst, folder, n_samples, n_steps):
    for i, dico in enumerate([best, worst]):
        if i == 0:
            prefix = "best"
        else:
            prefix = "worst"

        # Load clean reference data for plotting
        with open(dico["dataset_path"], "rb") as f:
            ref_data = pkl.load(f)
        X_ref = ref_data["X"]
        corruption_type = ref_data["type"]
        mode = ref_data.get("mode", "2D")
        
        method = dico["method"]
        p = dico["p"]
        delta = dico["delta"]
        module = dico["module"]
        further_corrupter = dico["corrupter"]
        noise_scheduler = dico["scheduler"]

        os.makedirs(str(folder / "viz"), exist_ok=True)
        filename = folder / "viz" / f"{prefix}_sampling_{method}_{p:.4f}_{delta:.4f}.gif"

        # Generate sampling GIF with reference overlay
        LOGGER.info(f"Generating {prefix} sampling GIF...")
        if corruption_type in ["inpainting", "gaussian", "inpainting_pw"]:
            if mode == "Nx2D":
                viz_sample_Nx2D(
                    module, noise_scheduler, further_corrupter,
                    n_clouds=9, n_points=ref_data.get("n_points_per_cloud", 200),
                    n_steps=n_steps,
                    output_path=str(filename),
                    ref_data=X_ref[:9],
                )
            else:
                viz_sample_2D(
                    module, noise_scheduler, further_corrupter,
                    n_samples=n_samples, n_steps=n_steps,
                    output_path=str(filename),
                    ref_data=X_ref,
                )


def load_config(path: str | Path):
    if isinstance(path, Path):
        path = str(path)
    
    with open(path, "r") as f:
        config = json.load(f)
    
    return config

def make_seeds(n):
    return np.random.randint(0,1000001, n)

def make_inpainting_datasets(datasets_cfg, dataset_type, folder):
    inpainting_type = datasets_cfg["type"]
    prevent_zero = datasets_cfg.get("prevent_zero", True)
    p_list = datasets_cfg["p"]
    mode = datasets_cfg.get("mode", "2D")
    X_params = datasets_cfg["X_params"][dataset_type]

    dataset_folder = folder / "datasets"
    os.makedirs(str(dataset_folder), exist_ok=True)
    
    traces = []
    
    if mode == "2D":
        if inpainting_type == "inpainting":
            func = inpainting_corruption
            kwargs = {"prevent_zero": prevent_zero}
        elif inpainting_type == "inpainting_pw":
            func = inpainting_corruption_pointwise
            kwargs = {}
    elif mode == "Nx2D":
        if inpainting_type == "inpainting":
            func = inpainting_corruption_Nx2D
            kwargs = {"prevent_zero": prevent_zero}
        elif inpainting_type == "inpainting_pw":
            func = inpainting_corruption_pointwise_Nx2D
            kwargs = {}

    for p in p_list:
        for seed in SEEDS:
            X_params["seed"] = seed
            
            if mode == "2D":
                X = generate_data(dataset_type, **X_params)
            else:
                X = generate_Nx2D_data(dataset_type, **X_params)

            kwargs["p"] = p
            rng = np.random.default_rng(seed + 1)
            kwargs["rng"] = rng
            Y, A = func(X, **kwargs)

            # Save
            filename = dataset_folder / f"{dataset_type}_{p}_{seed}.pkl"
            
            data = {"mode": mode, "type": inpainting_type, "X": X, "A": A}
            if mode == "Nx2D":
                data["n_points_per_cloud"] = X_params.get("n_points_per_cloud", 200)

            with open(str(filename), "wb") as f:
                pkl.dump(data, f)
            
            traces.append((str(filename), p))
    return traces

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to JSON config file")
    
    args = parser.parse_args()

    print("Loading config..")
    path_cfg = Path(args.cfg).resolve()
    assert str(path_cfg).endswith(".json"), f"Config file must be a JSON but got {path_cfg}"

    cfg_dict = load_config(path_cfg)

    # Setting seeds
    n_seeds = cfg_dict["n_replicates"]
    global SEEDS
    SEEDS = make_seeds(n_seeds)
    
    # identifying metric for benchmark
    cfg_metric_list = cfg_dict["metrics"]
    metric_list = []
    for i, m in enumerate(cfg_metric_list):
        if m.lower() in METRIC_DICT.keys():
            metric_list.append(m.lower())
        else:
            print(f"Metric {m} is unknown and will be skipped.")
    assert len(metric_list) >= 1, f"All user defined metrics were unknown. Use at least one of {list(METRIC_DICT.keys())}."
    ranking_metric = metric_list[0]

    datasets_cfg = cfg_dict["datasets"]
    inpainting_type = datasets_cfg["type"]
    mode = datasets_cfg.get("mode", "2D")
    delta_list = datasets_cfg["delta"]

    # Making output_folder
    folder = cfg_dict["output_folder"]
    folder = "/".join(folder.split("/")[:-1]) + f"/{inpainting_type}_{mode}_results"
    OUTPUT_FOLDER = Path(folder).resolve()
    os.makedirs(str(OUTPUT_FOLDER), exist_ok=True)

    logging.basicConfig(filename=str(OUTPUT_FOLDER / "log.txt"),
                        encoding = "utf-8",
                        filemode = "a",
                        format = "{asctime} - {levelname} - {message}",
                        style = "{",
                        datefmt = "%Y-%m-%d %H:%M",
                        level = logging.INFO
                    )
    global LOGGER
    LOGGER = logging.getLogger()

    training_cfg = cfg_dict["training"]
    device = torch_device("cuda" if cuda_is_available() else "cpu")
    LOGGER.info(f"Using {device}")
    training_cfg["device"] = device

    viz_cfg = cfg_dict["viz"]

    for dataset_type in ["two_moons", "swiss_roll"]:

        LOGGER.info(f"Making {dataset_type.upper()} datasets..")
        datasets = make_inpainting_datasets(datasets_cfg, dataset_type, OUTPUT_FOLDER)

        results_metrics_list = []
        best_metric = np.inf
        worst_metric = -np.inf

        for dataset_path, p in datasets:
            LOGGER.info("="*80)
            LOGGER.info(str(dataset_path).upper())
            LOGGER.info("="*80)
            
            LOGGER.info("Launching experiments..")
            results_metrics, best, worst, loss_curves = launch_experiments(dataset_path, p, delta_list, 
                                                                           metric_list, ranking_metric, 
                                                                           **training_cfg)

            results_metrics_list.extend(results_metrics)
            
            if best["metrics"][ranking_metric] < best_metric:
                best_overall = best
                best_metric = best["metrics"][ranking_metric]

            if worst["metrics"][ranking_metric] > worst_metric:
                worst_overall = worst
                worst_metric = worst["metrics"][ranking_metric]

        LOGGER.info("Saving results table..")
        make_table(results_metrics_list, metric_list, OUTPUT_FOLDER)

        LOGGER.info("Making visualizations..")
        plot_loss_curves(loss_curves, OUTPUT_FOLDER)

        visualize_best_worse(best_overall, worst_overall, OUTPUT_FOLDER, 
                            **viz_cfg)

        LOGGER.info(f"All {dataset_type} experiments completed.")

if __name__ == "__main__":
    main()