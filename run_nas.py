import nni.retiarii.strategy as strategy
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nas.learning_utils import evaluate_model, evaluate_model_darts
from nas.estimator import HardwareMetricFilter
from nas.model import MLPSpace, ResNetSpace, FTTransformerSpace

import logging
import argparse
import os
_logger = logging.getLogger(__name__)
os.environ['PICKLE_SIZE_LIMIT'] = str(10*1024*1024*1024)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Just an example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--strategy", type=str,
                        help="NAS search stragegy, currently, only supported for random, evolution, and reinforce strategies", default="random")
    parser.add_argument("-r", "--lag_range", type=int,
                        help="Number of lag features. MISO dataset is a time-series dataset. Therefore, we also use\
                            some lag features from previos time lags to forecast for a future label. E.g. 26 meaning \
                            that we use features of the last 26 timesteps", default=26)
    parser.add_argument("-n", "--trial_number", type=int,
                        help="Maximum number of trial samples", default=20)
    parser.add_argument("-c", "--n_gpus", type=int,
                        help="Number of gpus, you want to run NAS", default=0)
    parser.add_argument("-p", "--port", type=int,
                        help="NNI WebUI Port. You can go to https://web-IP:port to access NNI WebUI", default=8081)
    parser.add_argument("-t", "--target", type=str,
                        help="Training Label Name. This option depends on the dataset. We currently support MISO dataset only. \
                        There are two targets for MISO dataset, namely ref_ch4(ppm) and ref_h2o(ppm)", default="ref_ch4(ppm)")
    parser.add_argument("-me", "--metrics", type=list,
                        help="The Optimized Metric. We support 4 metrics: MPAE, energy, latency and MSE", default=["MPAE", "energy"])
    parser.add_argument("-m", "--mode", type=str,
                        help="Constraint or MOO mode. We support three modes, namely debug, mmo and filter. \
                        Debug means that we run NAS with accuracy only. MMO mean that NAS with energy and \
                        accuracy. Filter meaning that we apply a filter for the energy", default="debug")
    parser.add_argument("-th", "--target_hardware", type=str,
                        help="Target hardware. We support 2 platforms: myriadvpu_openvino2019r2 and jetsonnano_jetpack46", default="myriadvpu_openvino2019r2")
    parser.add_argument("-bm", "--backbone_model", type=str,
                        help="The base model of NAS. We support 3 backbone models: mlp, resnet, and fttransformer", default="mlp")

    thresholds = {"latency": 5, "accuracy": 0.3}

    args = parser.parse_args()
    cfg = vars(args)
    if "energy" in cfg['metrics'] and "latency" in cfg['metrics']:
        cfg['metrics'].remove("latency")
    if cfg['backbone_model'] == 'mlp':
        model_space = MLPSpace(n_features=cfg['lag_range']+7)
    elif cfg['backbone_model'] == 'resnet':
        model_space = ResNetSpace(n_features=cfg['lag_range']+7)
    elif cfg['backbone_model'] == 'fttransformer':
        model_space = FTTransformerSpace(n_features=cfg['lag_range']+7)
    else:
        raise Exception('Not support this backbone model yet!')

    evaluator = FunctionalEvaluator(evaluate_model, lag_range=cfg['lag_range'],
                                    target=cfg['target'],
                                    optimized_metrics=cfg['metrics'],
                                    mode=cfg['mode'],
                                    target_values=thresholds)

    if cfg['mode'] == "filter":
        model_filter = HardwareMetricFilter(
            thresholds, applied_hardware=cfg['target_hardware'], reverse=False)
    else:
        model_filter = None

    if cfg["strategy"] == "random":
        search_strategy = strategy.Random(
            dedup=True, model_filter=model_filter)
    elif cfg["strategy"] == "evolution":
        search_strategy = strategy.RegularizedEvolution(optimize_mode="maximize",
                                                        sample_size=cfg["trial_number"]//8,
                                                        population_size=cfg["trial_number"]//2,
                                                        cycles=cfg["trial_number"],
                                                        model_filter=model_filter
                                                        )
    elif cfg["strategy"] == "reinforce":
        if cfg["trial_number"] >= 20:
            search_strategy = strategy.PolicyBasedRL(
                max_collect=cfg["trial_number"]//20, trial_per_collect=20)
        else:
            search_strategy = strategy.PolicyBasedRL(
                max_collect=cfg["trial_number"]//2, trial_per_collect=2)
    elif cfg["strategy"] == "darts":
        search_strategy = strategy.DARTS()
        evaluator = evaluate_model_darts(
            lag_range=cfg['lag_range'], target=cfg['target'], n_gpus=cfg['n_gpus'],
            max_epochs=50, fast_dev_run=False)

    elif cfg["strategy"] == "moo_evolution":
        search_strategy = strategy.MultiObjectiveRegularizedEvolution(
            sample_size=cfg["trial_number"]//8,
            population_size=cfg["trial_number"]//2,
            cycles=cfg["trial_number"],
            model_filter=model_filter
        )
    assert (cfg['mode'] == 'moo_v2' and cfg["strategy"] == "moo_evolution") or \
        (cfg["mode"] != "moo_v2" and cfg["strategy"] != "moo_evolution")

    exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'mnist_search'
    exp_config.execution_engine = 'base' if cfg["strategy"] != "darts" else 'oneshot'
    # spawn 4 trials at most
    exp_config.max_trial_number = cfg["trial_number"]
    exp_config.experiment_working_directory = "./nni-experiments/"
    if cfg["n_gpus"] > 0:
        # will run two trials concurrently
        exp_config.trial_concurrency = cfg["n_gpus"]
        exp_config.trial_gpu_number = cfg["n_gpus"]
        exp_config.training_service.use_active_gpu = True
    else:
        exp_config.trial_concurrency = 1  # will run two trials concurrently
        exp_config.trial_gpu_number = 0
        exp_config.training_service.use_active_gpu = False

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if "ch4" in cfg['target']:
        trimmed_target = "ch4"
    else:
        trimmed_target = "h2o"
    folder_path = 'results/{}_{}_{}_{}_{}'.format(
        cfg["strategy"], trimmed_target, "-".join(cfg['metrics']), cfg["trial_number"], cfg['backbone_model'])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Start running")
    exp.run(exp_config, port=cfg["port"])
    print("Done NAS!!!")
    print("Start to write")

    for idx, model_code in enumerate(exp.export_top_models(top_k=10, formatter="code")):
        file_path = os.path.join(folder_path, "top_{}.py".format(idx + 1))
        with open(file_path, 'w') as f:
            f.write(model_code)

    for idx, model_code in enumerate(exp.export_top_models(top_k=10, optimize_mode="minimize", formatter="code")):
        file_path = os.path.join(folder_path, "bottom_{}.py".format(idx + 1))
        with open(file_path, 'w') as f:
            f.write(model_code)

    # for idx, model_code in enumerate(exp.export_top_models(top_k=1, formatter="dict")):
    #     print(model_code)
        # file_path = os.path.join(folder_path, "top_{}_dict.py".format(idx + 1))
        # with open(file_path, 'w') as f:
        #     f.write(model_code)

    print("Done!!!")
