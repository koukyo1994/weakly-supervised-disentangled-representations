import json
import os

from .evaluate import evaluate_with_gin

from disentanglement_lib.config.unsupervised_study_v1 import sweep as unsupervised_study_v1


def compute_metrics(exp_path,
                    dataset_name,
                    random_seed):
    overwrite = True
    _study = unsupervised_study_v1.UnsupervisedStudyV1()
    evaluation_configs = sorted(_study.get_eval_config_files())

    expected_evaluation_metrics = [
        "factor_vae_metric",
        "modularity_explicitness",
        "sap_score",
        "mig"
    ]

    for gin_eval_config in evaluation_configs:
        metric_name = gin_eval_config.split("/")[-1].replace(".gin", "")
        if metric_name not in expected_evaluation_metrics:
            continue
        print(f"Evaluationg Metric: {metric_name}")
        result_path = os.path.join(exp_path, "metrics", metric_name)
        representation_path = os.path.join(exp_path, "representations")
        eval_bindings = [
            f"evaluation.name = '{metric_name}'"
        ]
        evaluate_with_gin(
            representation_path,
            result_path,
            dataset_name,
            overwrite,
            [gin_eval_config],
            random_seed,
            eval_bindings)

    # Gather evaluation results
    evaluation_result_template = "{}/metrics/{}/results/aggregate/evaluation.json"
    final_scores = {}
    for _metric_name in expected_evaluation_metrics:
        evaluation_json_path = evaluation_result_template.format(
            exp_path,
            _metric_name
        )
        evaluation_results = json.loads(
                open(evaluation_json_path, "r").read()
        )

        if _metric_name == "factor_vae_metric":
            _score = evaluation_results["evaluation_results.eval_accuracy"]
            final_scores["factor_vae_metric"] = _score
        elif _metric_name == "beta_vae_sklearn":
            _score = evaluation_results["evaluation_results.eval_accuracy"]
            final_scores["beta_vae_metric"] = _score
        elif _metric_name == "modularity_explicitness":
            _score = evaluation_results["evaluation_results.modularity_score"]
            final_scores["modularity_score"] = _score
        elif _metric_name == "dci":
            _score = evaluation_results["evaluation_results.disentanglement"]
            final_scores["dci"] = _score
        elif _metric_name == "mig":
            _score = evaluation_results["evaluation_results.discrete_mig"]
            final_scores["mig"] = _score
        elif _metric_name == "sap_score":
            _score = evaluation_results["evaluation_results.SAP_score"]
            final_scores["sap_score"] = _score
        elif _metric_name == "irs":
            _score = evaluation_results["evaluation_results.IRS"]
            final_scores["irs"] = _score
        else:
            raise Exception("Unknown metric name : {}".format(_metric_name))

    with open(f'{exp_path}/metric_results.json', 'w') as json_file:
        json.dump(final_scores, json_file)
    print("Final Scores : ", final_scores)