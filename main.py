import contextlib
import os
import sys
import warnings
import webbrowser
from collections import Counter, defaultdict
from itertools import product
from time import monotonic
from typing import Dict, Tuple, Optional  # mainly to resolve erroneous PyCharm type warnings

import numpy as np  # pip install numpy==2.2.6, last version compatible with Python 3.10
import pandas as pd
import joblib
import altair as alt
from tqdm import tqdm

from sklearn.svm import LinearSVC  # pip install scikit-learn==1.7.2, last version compatible with Python 3.10
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows
# Python 3.10 is the last Python release supported by TensorFlow 2.10
# But
# Newer versions of Python, Numpy, and Sklearn shouldn't require code changes, except for LogisticRegression.


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager for patching tqdm/joblib to display completion progress instead of the job queue"""
    # https://stackoverflow.com/a/58936697/16963475
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def bayesian_bootstrap_means_and_dominance(model_scores, rope, n_bootstrap_draws, random_state, credible_mass, only_means):
    names = list(model_scores.keys())
    per_model_scores: Dict[str, np.ndarray] = {}
    for k in names:
        per_model_scores[k] = np.asarray(model_scores[k], dtype=float)

    for a in per_model_scores.values():
        if a.ndim != 1:
            raise ValueError("All inputs must be 1D arrays")

    n_observations_per_model = []
    for a in per_model_scores.values():
        n_observations_per_model.append(len(a))

    if len(set(n_observations_per_model)) != 1:
        raise ValueError(
            "Joint Dirichlet resampling requires all score arrays to have the same length. "
            f"Got lengths: {dict(zip(names, n_observations_per_model))}"
        )

    n_observations = n_observations_per_model[0]
    rng = np.random.RandomState(random_state)

    # Computation of marginal mean posteriors giving stable estimates of each model's expected performance.
    # This is not affected by cross-model correlation.
    mean_posterior_samples = {}
    for name in tqdm(names, desc="Bayes Means", file=sys.stdout):
        a = per_model_scores[name]
        weights = rng.dirichlet(np.ones(n_observations), n_bootstrap_draws)
        mean_posterior_samples[name] = np.matmul(weights, a)

    alpha = (1.0 - credible_mass) / 2.0
    lo, hi = alpha, 1.0 - alpha

    marginal_mean_summaries: Dict[str, Tuple[float, float, float]] = {}
    for k in names:
        values: np.ndarray = mean_posterior_samples[k]
        mean = float(values.mean())
        ci_lo = float(np.quantile(values, lo))
        ci_hi = float(np.quantile(values, hi))
        marginal_mean_summaries[k] = (mean, ci_lo, ci_hi)

    practically_best_probs = None

    if not only_means:
        if rope is None:
            rope_list = [0]
        else:
            rope_list = rope if isinstance(rope, (list, tuple)) else [rope]

        # Joint Dirichlet to estimate probability of being "practically best" under a ROPE-soft argmax.
        # This does account for cross-model correlation.
        probs_per_rope: Dict[float, np.ndarray] = joint_practical_best_probs(
            rng=rng,
            score_dict=per_model_scores,
            rope_list=rope_list,
            n_observations=n_observations,
            total_draws=n_bootstrap_draws * 20,
            batch_size=1000,
        )

        practically_best_probs: Optional[Dict[float, Dict[str, float]]] = {}
        for r, prob_array in probs_per_rope.items():
            practically_best_probs[r] = dict(zip(names, map(float, prob_array)))

    return marginal_mean_summaries, practically_best_probs


def joint_practical_best_probs(rng, score_dict, rope_list, n_observations, total_draws, batch_size):
    names = list(score_dict.keys())
    score_arrays = [score_dict[name] for name in names]
    n_models = len(score_arrays)

    accumulated_soft_wins: Dict[float, np.ndarray] = {
        r: np.zeros(n_models, dtype=np.float64) for r in rope_list
    }
    draws_seen = 0

    with tqdm(total=total_draws, desc="Bayes Bests", file=sys.stdout) as pbar:
        prev = draws_seen
        while draws_seen < total_draws:
            B = min(batch_size, total_draws - draws_seen)

            weights = rng.dirichlet(np.ones(n_observations), B)

            joint_means = np.column_stack([
                np.matmul(weights, a) for a in score_arrays
            ])

            row_wise_best_mean = joint_means.max(axis=1, keepdims=True)
            gap_from_best = row_wise_best_mean - joint_means

            # Apply each rope independently to the same gap matrix. Thus
            # the expensive part (sampling Dirichlet weights and computing joint means) only happens once
            for r in rope_list:
                rope_softening_scale = 0.10 * r if r > 0 else 10**-6
                # Controls how sharply the ROPE threshold transitions from winner-takes-all (rope_softening_scale =~ 0) to shared credit
                # Smooth approximation instead of hard ROPE threshold to mostly reduce Monte Carlo tie-breaking noise
                # + Softening scale is proportional to ROPE to retain scale invariance
                rope_margin_logit = (gap_from_best - r) / rope_softening_scale
                np.clip(rope_margin_logit, -40, 40, out=rope_margin_logit)
                soft_win_weights = 1.0 / (1.0 + np.exp(rope_margin_logit))
                soft_win_weights /= soft_win_weights.sum(axis=1, keepdims=True)
                accumulated_soft_wins[r] += soft_win_weights.sum(axis=0)

            draws_seen += B
            delta = draws_seen - prev
            pbar.update(delta)
            prev = draws_seen

    probs_per_rope = {}
    for r in rope_list:
        probs_per_rope[r] = accumulated_soft_wins[r] / draws_seen
    return probs_per_rope


def report_bayesian_bootstrap_results(model_scores, rope, n_bootstrap_draws, random_state, credible_mass, top_k_per_prefix, only_means, top_r_practically_best):
    marginal_summaries, practically_best_probs = bayesian_bootstrap_means_and_dominance(
        model_scores=model_scores,
        rope=rope,
        n_bootstrap_draws=n_bootstrap_draws,
        random_state=random_state,
        credible_mass=credible_mass,
        only_means=only_means
    )

    credible_interval_label = f"{int(credible_mass * 100)}%"

    print()
    print(f"=== Bayesian Bootstrap Dominance ===")
    print(f"  Posterior Youden / Informedness means & {credible_interval_label} credible int")
    print(f"  [independent per-model Dirichlet]:")

    def mean_from_summary(item):
        _, (mean_value, _, _) = item
        return mean_value

    ranked = sorted(
        marginal_summaries.items(),
        key=mean_from_summary,
        reverse=True,
    )

    seen_prefixes = {}
    for name, _ in ranked:
        prefix = name.split("_")[0]
        if prefix not in seen_prefixes:
            seen_prefixes[prefix] = []
        seen_prefixes[prefix].append(name)

    top_names_by_prefix = {}
    for prefix, names in seen_prefixes.items():
        top_names_by_prefix[prefix] = names[:top_k_per_prefix]

    all_labels = []
    for names in top_names_by_prefix.values():
        for name in names:
            all_labels.append(f"    M({name})")
    if practically_best_probs is not None:
        for r in (rope if isinstance(rope, (list, tuple)) else [rope]):
            for names in top_names_by_prefix.values():
                for name in names:
                    all_labels.append(f"    P(rope={r}, {name})")

    column_width = max(len(s) for s in all_labels)

    def pad(label):
        return label + " " * (column_width - len(label))

    for i, (prefix, top_names) in enumerate(top_names_by_prefix.items()):
        if i > 0:
            print("    ...")
        for name in top_names:
            mean, lo, hi = marginal_summaries[name]
            print(f"{pad(f'    M({name})')}  = {mean:.4f}  CI [{lo:.4f}, {hi:.4f}]")

    if practically_best_probs is not None:
        rope_list = rope if isinstance(rope, (list, tuple)) else [rope]
        for r in rope_list:
            print()
            print(f"  Top {top_r_practically_best} ROPE-soft dominance probabilities [rope=±{r}, joint Dirichlet]:")
            print(f"  (probability a given model is practically best)")

            for name, prob in sorted(
                    practically_best_probs[r].items(),
                    key=lambda x: x[1],
                    reverse=True
            )[:top_r_practically_best]:
                print(f"{pad(f'    P(rope={r}, {name})')}  = {prob:.4f}")


def process_split(train_idx, test_idx, x, y, classifier_pipelines):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # LinearSVC might fail to converge especially for extreme C-Values
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    youden_local = {}

    for name, pipeline in classifier_pipelines.items():
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        true_positive_rate = recall_score(y_test, y_pred, average="binary", pos_label=0)  # malignant = 0
        # in case of multi-class classification: true_positive_rate = recall_score(y_test, y_pred, average="macro")
        true_negative_rate = recall_score(y_test, y_pred, average="binary", pos_label=1)  # benign = 1
        false_positive_rate = 1 - true_negative_rate

        youden = true_positive_rate - false_positive_rate
        # in case of multi-class classification: youden = 1/(n-1) * (n*macroTPR - 1), with n being the number of classes
        # first described as such by Sébastien Foulle in:
        # "Mathematical Characterization of Better-than-Random Multiclass Models"

        youden_local[name] = youden

    return youden_local


def run_cv_parallel(x, y, classifier_pipelines, outer_cv, n_jobs):
    total_splits = outer_cv.get_n_splits()

    with tqdm_joblib(tqdm(total=total_splits, desc="CV Progress", smoothing=0.05, file=sys.stdout)):
        results = joblib.Parallel(n_jobs=n_jobs, batch_size=1)(
            joblib.delayed(process_split)(
                train_idx, test_idx, x, y, classifier_pipelines
            )
            for train_idx, test_idx in outer_cv.split(x, y)
        )

    youden_scores_by_model = defaultdict(list)

    for youden_local in results:
        for name in youden_local:
            youden_scores_by_model[name].append(youden_local[name])

    return youden_scores_by_model


def main():
    start_time = monotonic()

    data = load_breast_cancer()
    x_df = pd.DataFrame(data.data, columns=data.feature_names)

    # -----------------------

    results = []

    for col in x_df.columns:

        types_present = x_df[col].map(type).unique()
        type_names = [t.__name__ for t in types_present]

        missing_values = x_df[col].isna().sum()

        row = {
            "feature": col,
            "types_present": type_names,
            "missing": missing_values
        }

        results.append(row)

    results_df = pd.DataFrame(results)

    pd.set_option('display.max_columns', None)
    print(results_df)

    # -----------------------

    y2 = pd.Series(data.target, name='class')
    df = pd.concat([x_df, y2], axis=1)

    df.columns = [c.replace(' ', '_') for c in df.columns]
    features = [c for c in df.columns if c != 'class']
    classes = sorted(df['class'].unique())

    row_charts = []

    for cls in classes:
        df_cls = df[df['class'] == cls]

        label = alt.Chart(pd.DataFrame({'class_label': [f"Class {cls}"]})).mark_text(
            align='center',
            baseline='middle',
            fontSize=14,
            dx=5
        ).encode(
            y=alt.value(300 / 2),
            text='class_label:N'
        ).properties(width=100, height=300)

        hist_charts = []
        for feature in features:
            n = len(df_cls)
            num_bins = int(np.ceil((2 * n) ** (1 / 3)))  # oversmoothed/Terrell-Scott rule, 1985
            min_val = df_cls[feature].min()
            max_val = df_cls[feature].max()
            bin_width = (max_val - min_val) / num_bins

            hist = alt.Chart(df_cls).mark_bar().encode(
                x=alt.X(feature, type='quantitative', bin=alt.Bin(step=bin_width)),
                y=alt.Y('count()', axis=None)
            ).properties(width=300, height=300)

            hist_charts.append(hist)

        row = alt.hconcat(label, *hist_charts, spacing=5)
        row_charts.append(row)

    final_chart = alt.vconcat(*row_charts, spacing=10).resolve_scale(x='independent', y='independent')
    final_chart.save('histograms_by_class.html')
    webbrowser.open('file://' + os.path.realpath('histograms_by_class.html'))

    # -----------------------

    N_REPEATED_RUNS = 250
    N_CV_FOLDS = 4
    N_CPU_THREADS = 3

    # -----------------------

    cancer = load_breast_cancer()
    x = cancer.data
    y = cancer.target
    class_names = cancer.target_names

    print()
    print("=== Dataset Prevalence ===")
    class_counts = Counter(y)
    n_total_samples = len(y)

    for class_id, count in class_counts.items():
        prevalence = count / n_total_samples
        print(f"Class '{class_names[class_id]}': {prevalence:.4f} ({count}/{n_total_samples})")
    print()

    # 80 linearSVCs
    c_exponents = []
    for i in range(-10, 10):
        current = round(2 ** i, 5)
        lower = round(2 ** (i - 1), 5)
        c_exponents.append(current)
        c_exponents.append(round(current + lower, 5))
    loss_values = ["hinge", "squared_hinge"]
    svc_classifiers = {
        f"svc_C{C}_{loss}": LinearSVC(
            C=C,
            loss=loss,
            max_iter=10000
        )
        for C, loss in product(
            c_exponents,
            loss_values
        )
    }

    # L1 logistic regressions i.e. liblinear & saga do NOT lead to general agreement between CV subset and full dataset:
    # Because L1 regularization is highly sensitive to feature scales i.e.
    # Small changes in scale or dataset size → different sparsity patterns → drastically different predictions.

    # 60 L2 logistic regressions
    start = 0.01
    end = 10
    n = 20
    r = (end / start) ** (1 / (n - 1))
    c_values = [round(start * r ** i, 5) for i in range(n)]
    # "sag", "saga" and "liblinear" are non deterministic
    solver_values = ["lbfgs", "newton-cg", "newton-cholesky"]

    lr_classifiers = {
        f"l2_lr_C{C}_{solver}": LogisticRegression(
            C=C,
            penalty="l2",
            # l1_ratio=0,  # sklearn 1.8.0 and beyond: replace penalty
            solver=solver,
            max_iter=1000
        )
        for C, solver in product(
            c_values,
            solver_values
        )
    }

    classifiers = {**svc_classifiers, **lr_classifiers}

    classifier_pipelines = {
        name: Pipeline([("scaler", StandardScaler()), (name, clf)])
        for name, clf in classifiers.items()
    }

    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_CV_FOLDS,
        n_repeats=N_REPEATED_RUNS,
        random_state=None
    )

    youden_scores_by_model = run_cv_parallel(
        x=x,
        y=y,
        classifier_pipelines=classifier_pipelines,
        outer_cv=outer_cv,
        n_jobs=N_CPU_THREADS
    )

    print("\n")
    report_bayesian_bootstrap_results(
        model_scores=youden_scores_by_model,
        rope=None,
        n_bootstrap_draws=25000,
        random_state=None,
        credible_mass=0.99,
        top_k_per_prefix=6,
        only_means=True,
        top_r_practically_best=6
    )

    print()

    # --------------------------------------------------------

    svc_c = 0.01172
    svc_loss = "squared_hinge"
    lr_c = 0.54556
    lr_solver = "lbfgs"

    classifiers = {
        "svc_vanilla": LinearSVC(),
        "svc_tuned": LinearSVC(C=svc_c, loss=svc_loss),
        "lr_vanilla": LogisticRegression(),
        "lr_tuned": LogisticRegression(solver=lr_solver, C=lr_c, penalty="l2")
    }

    classifier_pipelines = {
        name: Pipeline([("scaler", StandardScaler()), (name, clf)])
        for name, clf in classifiers.items()
    }

    outer_cv = RepeatedStratifiedKFold(
        n_splits=N_CV_FOLDS,
        n_repeats=N_REPEATED_RUNS,
        random_state=None
    )

    youden_scores_by_model = run_cv_parallel(
        x=x,
        y=y,
        classifier_pipelines=classifier_pipelines,
        outer_cv=outer_cv,
        n_jobs=N_CPU_THREADS
    )

    print("\n")
    report_bayesian_bootstrap_results(
        model_scores=youden_scores_by_model,
        rope=[0.005, 0.01, 0.015],  # rope can be a single float or a list of floats
        n_bootstrap_draws=25000,
        random_state=None,
        credible_mass=0.99,
        top_k_per_prefix=6,
        only_means=False,
        top_r_practically_best=6
    )

    # --------------------------------------------------------

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(solver=lr_solver, C=lr_c, penalty="l2")
    model.fit(x_scaled, y)
    weights = model.coef_[0]
    bias = model.intercept_[0]
    feature_names = [f for f in cancer.feature_names]
    terms = []
    for wi, name in zip(weights, feature_names):
        sign = "+" if wi >= 0 else "-"
        terms.append(f" {sign} {abs(wi):.4f}·{name}")
    equation = "".join(terms) + f" {('+' if bias >= 0 else '-')} {abs(bias):.4f}"
    print("\nLR:  f(x) =" + equation)
    print()
    feature_impacts = sorted(zip(weights, feature_names), key=lambda z: abs(z[0]), reverse=True)
    print("Feature impact (high → low):")
    for wi, name in feature_impacts:
        sign = "+" if wi >= 0 else "-"
        print(f"  {sign}{abs(wi):.4f}  {name}")
    print()

    scores_all = np.matmul(x_scaled, weights) + bias
    y_pred = (scores_all > 0).astype(int)
    cm1 = confusion_matrix(y, y_pred, labels=[1, 0])  # Class 0 aka Malignant == positive, Class 1 aka Benign == negative
    # [[TN FP]
    #  [FN TP]]
    print(cm1, "\n")

    false_negative_mask = (y == 0) & (y_pred == 1)
    false_negative_indices = np.argwhere(false_negative_mask).flatten()
    print(f"Number of False Negatives: {len(false_negative_indices)}")
    print(f"Indices: {false_negative_indices}\n")
    print("=== False Negative Characteristics ===")
    fn_df = x_df.iloc[false_negative_indices, 0:5].copy()  # first 5 features
    fn_df["score"] = scores_all[false_negative_indices]
    print(fn_df, "\n")

    # --------------------------------------------------------

    model = LinearSVC(C=svc_c, loss=svc_loss)
    # dual cant be False when loss=hinge, which leads to tiny nondeterminism
    model.fit(x_scaled, y)
    weights = model.coef_[0]
    bias = model.intercept_[0]

    feature_names = [f for f in cancer.feature_names]
    terms = []
    for wi, name in zip(weights, feature_names):
        sign = "+" if wi >= 0 else "-"
        terms.append(f" {sign} {abs(wi):.4f}·{name}")
    equation = "".join(terms) + f" {('+' if bias >= 0 else '-')} {abs(bias):.4f}"
    print("SVC: f(x) =" + equation)
    print()
    feature_impacts = sorted(zip(weights, feature_names), key=lambda z: abs(z[0]), reverse=True)
    print("Feature impact (high → low):")
    for wi, name in feature_impacts:
        sign = "+" if wi >= 0 else "-"
        print(f"  {sign}{abs(wi):.4f}  {name}")
    print()

    scores_all = np.matmul(x_scaled, weights) + bias
    y_pred = (scores_all > 0).astype(int)
    cm2 = confusion_matrix(y, y_pred, labels=[1, 0])  # Class 0 aka Malignant == positive, Class 1 aka Benign == negative
    print(cm2, "\n")
    # [[TN FP]
    #  [FN TP]]

    false_negative_mask = (y == 0) & (y_pred == 1)
    false_negative_indices = np.argwhere(false_negative_mask).flatten()
    print("=== False Negatives ===")
    fn_df = x_df.iloc[false_negative_indices, 0:0].copy()
    fn_df["score"] = scores_all[false_negative_indices]
    print(fn_df)

    end_time = monotonic()
    print()
    print(f"\nTime elapsed: {end_time - start_time:.2f}s")


if __name__ == '__main__':
    main()
