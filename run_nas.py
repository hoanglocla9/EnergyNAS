import nni.retiarii.strategy as strategy
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nas.learning_utils import  get_regressor
from nas.model import MLPSpace, ResNetSpace, FTTransformerSpace, ConventionalTransformerSpace, FTTransformerSpace_OneShot

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
    parser.add_argument("-nf", "--n_features", type=int,
                        help="Number of features in dataset", default=0)
    parser.add_argument("-n", "--trial_number", type=int,
                        help="Maximum number of trial samples", default=20)
    parser.add_argument("-c", "--n_gpus", type=int,
                        help="Number of gpus, you want to run NAS", default=0)
    parser.add_argument("-p", "--port", type=int,
                        help="NNI WebUI Port. You can go to https://web-IP:port to access NNI WebUI", default=8081)
    parser.add_argument("-t", "--target", type=str,
                        help="Training Label Name. This option depends on the dataset. We currently support MISO dataset only. \
                        There are two targets for MISO dataset, namely ref_ch4(ppm) and ref_h2o(ppm)", default="ref_ch4(ppm)")
    parser.add_argument("-pm", "--performance_metric", type=str,
                        help="The Performance Metric. We support 2 metrics: MPAE, and MSE", default="MSE")
    parser.add_argument("-em", "--efficiency_metric", type=str,
                        help="The Efficiency Metric. We support 2 metrics: energy, and latency", default="energy")
    parser.add_argument("-m", "--mode", type=str,
                        help="Constraint or MOO mode. We support three modes, namely single and multi. \
                        Single means that we run NAS with accuracy only. Multi mean that NAS with energy (latency) and \
                        accuracy.", default="single")
    parser.add_argument("-th", "--target_hardware", type=str,
                        help="Target hardware. We support 2 platforms: myriadvpu_openvino2019r2 and jetsonnano_jetpack46", default="myriadvpu_openvino2019r2")
    parser.add_argument("-bm", "--backbone_model", type=str,
                        help="The base model of NAS. We support 3 backbone models: mlp, resnet, and fttransformer", default="mlp")
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="Batch Size", default=32)
    parser.add_argument("-mep", "--max_epoches", type=int,
                        help="Maximum Epoches", default=20)

    args = parser.parse_args()
    cfg = vars(args)


    # assert cfg['mode'] == 'multi' and cfg['strategy'] not in ["darts", "random_oneshot"], \
    #                 f"Not support the combination of '{cfg['mode']}' mode and '{cfg['strategy']}' strategy yet!!!"
    
    e_metric_folder_name = "power" if cfg['efficiency_metric'] == "energy" else cfg['efficiency_metric']
    assert os.path.exists(
        f"./predictors/{cfg['target_hardware']}/{e_metric_folder_name}/")

    if cfg['lag_range'] > 0 and cfg['n_features'] == 0:
        n_features = cfg['lag_range']+7
    else:
        n_features = cfg['n_features']

    thresholds = {cfg['efficiency_metric']: 5, cfg['performance_metric']: 0.3}
    mutation_hooks = []
    if cfg['backbone_model'] == 'mlp':
        model_space = MLPSpace(n_features=n_features)
    elif cfg['backbone_model'] == 'resnet':
        model_space = ResNetSpace(n_features=n_features)
    elif cfg['backbone_model'] == 'fttransformer':
        # , batch_size=cfg['batch_size']
        if cfg["strategy"] not in ["random_oneshot", 'darts']:
            model_space = FTTransformerSpace(n_features=n_features)
        else:
            model_space = FTTransformerSpace_OneShot(n_features=n_features)
            mutation_hooks = FTTransformerSpace_OneShot.get_extra_mutation_hooks()
    elif cfg['backbone_model'] == 'transformer':
        model_space = ConventionalTransformerSpace(
            n_features=n_features)
    else:
        raise Exception('Not support this backbone model yet!')

    evaluator = get_regressor(lag_range=cfg['lag_range'],
                                    target=cfg['target'],
                                    performance_metric=cfg['performance_metric'],
                                    efficiency_metric=cfg['efficiency_metric'],
                                    mode=cfg['mode'],
                                    target_values=thresholds, 
                                    batch_size=cfg['batch_size'],
                                    max_epochs=cfg['max_epoches'], 
                                    fast_dev_run=False,
                                    strategy=cfg["strategy"])

    if cfg["strategy"] == "random":
        search_strategy = strategy.Random(
            dedup=True)
    elif cfg["strategy"] == "evolution":
        if cfg['mode'] == 'single':
            search_strategy = strategy.RegularizedEvolution(optimize_mode="maximize",
                                                        sample_size=cfg["trial_number"]//8,
                                                        population_size=cfg["trial_number"]//2,
                                                        cycles=cfg["trial_number"]
                                                        )
        elif cfg['mode'] == 'multi':
            search_strategy = strategy.MultiObjectiveRegularizedEvolution(
                sample_size=cfg["trial_number"]//8,
                population_size=cfg["trial_number"]//2,
                cycles=cfg["trial_number"]
            )
    elif cfg["strategy"] == "reinforce":
        if cfg["trial_number"] >= 20:
            search_strategy = strategy.PolicyBasedRL(
                max_collect=cfg["trial_number"]//20, trial_per_collect=20)
        else:
            search_strategy = strategy.PolicyBasedRL(
                max_collect=cfg["trial_number"]//2, trial_per_collect=2)
    elif cfg["strategy"] == "darts":
        search_strategy = strategy.DARTS(mutation_hooks=mutation_hooks)
    elif cfg["strategy"] == "random_oneshot":
        assert (cfg['backbone_model'] == 'fttransformer' and cfg["strategy"] == "random_oneshot")  , "Only support Random One Shot with FTTransformer!!!"
        search_strategy = strategy.RandomOneShot(mutation_hooks=mutation_hooks)

    exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'mnist_search'
    exp_config.execution_engine = 'base' if cfg["strategy"] not in ["darts", "random_oneshot"] else 'oneshot'
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
    folder_path = 'results/{}_{}_{}-{}_{}_{}'.format(
        cfg["strategy"], trimmed_target, cfg['performance_metric'], cfg['efficiency_metric'], cfg["trial_number"], cfg['backbone_model'])
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
