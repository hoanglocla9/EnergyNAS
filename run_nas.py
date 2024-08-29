import nni.retiarii.strategy as strategy
from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from learning_utils import evaluate_model
from model import CalibrationModelSpace
from mutator import BlockMutator
import logging, argparse, os
_logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Just an example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--strategy", type=str, help="search stragegy, currently, only supported for random and evolution strategies", default="random")
    parser.add_argument("-r", "--lag_range", type=int, help="number of lag features", default=26)
    parser.add_argument("-n", "--trial_number", type=int, help="number of trial steps", default=500)
    parser.add_argument("-c", "--n_gpus", type=int, help="number of gpus", default=0)
    parser.add_argument("-p", "--port", type=int, help="NNI WebUI Port", default=8081)
    parser.add_argument("-t", "--target", type=str, help="Training Label Name", default="ref_ch4(ppm)")
    parser.add_argument("-m", "--metrics", type=str, help="The Optimized Metric", default=["MAE", "Latency"])

    args = parser.parse_args()
    cfg = vars(args)

    model_space = CalibrationModelSpace()

    evaluator = FunctionalEvaluator(evaluate_model, lag_range=cfg['lag_range'], 
                                        target=cfg['target'], 
                                        optimized_metric=cfg['metrics'], 
                                        target_latency=5) 
    

    if cfg["strategy"] == "random":
        search_strategy = strategy.Random(dedup=True)
    elif cfg["strategy"] == "evolution":
        search_strategy = strategy.RegularizedEvolution(optimize_mode="maximize", 
                                                        sample_size = cfg["trial_number"]//8, 
                                                        population_size=cfg["trial_number"]//2, 
                                                        cycles=cfg["trial_number"]
                                                    )
    elif cfg["strategy"] == "reinforce":
        if cfg["trial_number"] >= 20:
            search_strategy = strategy.PolicyBasedRL(max_collect=cfg["trial_number"]//20, trial_per_collect=20)
        else:
            search_strategy = strategy.PolicyBasedRL(max_collect=cfg["trial_number"]//2, trial_per_collect=2)


    applied_mutators = [
        BlockMutator('mutable_all')
    ]

    exp = RetiariiExperiment(model_space, evaluator, applied_mutators, search_strategy)
    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'mnist_search'
    exp_config.execution_engine = 'base'
    exp_config.max_trial_number = cfg["trial_number"]   # spawn 4 trials at most
    if cfg["n_gpus"] > 0:
        exp_config.trial_concurrency = cfg["n_gpus"]  # will run two trials concurrently
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
    folder_path = 'results/{}_{}_{}_{}'.format(cfg["strategy"], trimmed_target, "-".join(cfg['metrics']), cfg["trial_number"])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Start running")
    exp.run(exp_config, port=cfg["port"])
    print("Done NAS!!!")
    print("Start to write")
    for idx, model_code in enumerate(exp.export_top_models(top_k=50, formatter="code")):
        file_path = os.path.join(folder_path, "top_{}.py".format(idx + 1))
        with open(file_path, 'w') as f:
            f.write(model_code)
    print("Done!!!")
