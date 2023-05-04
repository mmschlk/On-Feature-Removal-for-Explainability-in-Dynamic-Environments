import random

import numpy as np
import pandas as pd
import river.metrics
from matplotlib import pyplot as plt
from river.stream import iter_pandas
from river.utils import Rolling

from river.tree import HoeffdingAdaptiveTreeClassifier
from river.metrics import Accuracy

from ixai.explainer import IncrementalPFI, IncrementalSage
from ixai.storage import TreeStorage
from ixai.imputer import TreeImputer

from mdi import MeanDecreaseImpurityExplainer


if __name__ == "__main__":

    # Get Data -------------------------------------------------------------------------------------
    N_SAMPLES = 20_000

    N_SAMPLES_STREAM = 10_000

    RANDOM_SEED: int = 42


    all = np.ones((4, 4)) * 5
    identity = np.diag(np.ones(4))

    x_1 = np.random.multivariate_normal([0, 0, 0, 0], identity, size=20_000)
    x_2 = np.random.multivariate_normal([0, 0, 0, 0], all, size=20_000)

    y_1 = x_1[:, 0] > 0
    y_1 = y_1.astype(int)
    y_1 = pd.Series(y_1)

    y_2 = x_2[:, 0] > 0
    y_2 = y_2.astype(int)
    y_2 = pd.Series(y_2)

    stream_1 = pd.DataFrame(x_1, columns=['x_0', 'x_1', 'x_2', 'x_3'])
    #stream_1['y'] = y_1
    stream_2 = pd.DataFrame(x_2, columns=['x_0', 'x_1', 'x_2', 'x_3'])
    #stream_2['y'] = y_2

    feature_names = ['x_0', 'x_1', 'x_2', 'x_3']
    cat_feature_names = []
    num_feature_names = feature_names

    #model = LogisticRegression()

    model = HoeffdingAdaptiveTreeClassifier(leaf_prediction="nba", seed=RANDOM_SEED)

    # Get imputer and explainers -------------------------------------------------------------------
    model_function = model.predict_one
    loss_metric = river.metrics.Accuracy()
    training_metric = Rolling(river.metrics.Accuracy(), window_size=1000)

    # explainer interventional ---------------------------------------------------------------------
    smoothing_alpha = 0.001
    fi_explainer_interventional = IncrementalPFI(
        model_function=model_function,
        loss_function=Accuracy(),
        feature_names=feature_names,
        n_inner_samples=1,
        smoothing_alpha=smoothing_alpha
    )

    sage_explainer_interventional = IncrementalSage(
        model_function=model_function,
        loss_function=Accuracy(),
        feature_names=feature_names,
        n_inner_samples=1,
        smoothing_alpha=smoothing_alpha
    )

    # explainer observational ----------------------------------------------------------------------
    storage_observational = TreeStorage(
        cat_feature_names=cat_feature_names, num_feature_names=num_feature_names
    )

    imputer_observational = TreeImputer(
        model_function=model_function,
        storage_object=storage_observational,
        use_storage=True,
        direct_predict_numeric=True
    )

    fi_explainer_observational = IncrementalPFI(
        model_function=model_function,
        loss_function=Accuracy(),
        feature_names=feature_names,
        n_inner_samples=1,
        smoothing_alpha=smoothing_alpha,
        storage=storage_observational,
        imputer=imputer_observational
    )

    sage_explainer_observational = IncrementalSage(
        model_function=model_function,
        loss_function=Accuracy(),
        feature_names=feature_names,
        n_inner_samples=1,
        smoothing_alpha=smoothing_alpha,
        storage=storage_observational,
        imputer=imputer_observational
    )

    mdi_explainer = MeanDecreaseImpurityExplainer(
        feature_names=feature_names,
        tree_classifier=model,
    )

    fi_interventional_values = []
    fi_observational_values = []
    sage_interventional_values = []
    sage_observational_values = []
    mdi_values = []
    split_values = []

    # Train model ----------------------------------------------------------------------------------
    for n, (x_i, y_i) in enumerate(iter_pandas(stream_2, y_2), start=1):

        y_pred = model.predict_one(x_i)
        training_metric.update(y_i, y_pred)

        model.learn_one(x_i, y_i)

        fi_explainer_interventional.explain_one(x_i, y_i)
        fi_explainer_observational.explain_one(x_i, y_i)

        sage_explainer_interventional.explain_one(x_i, y_i)
        sage_explainer_observational.explain_one(x_i, y_i)

        mdi_val, splits = mdi_explainer.explain_one()
        mdi_values.append(mdi_val)
        split_values.append(splits)

        fi_interventional_values.append(fi_explainer_interventional.importance_values)
        fi_observational_values.append(fi_explainer_observational.importance_values)
        sage_interventional_values.append(sage_explainer_interventional.importance_values)
        sage_observational_values.append(sage_explainer_observational.importance_values)

        if n % 1000 == 0:
            print(f"Trained on {n} samples")
            print(f"Training accuracy: {training_metric.get():.3f}")
            print(f"pfi_interventional: {fi_explainer_interventional.importance_values}")
            print(f"pfi_observational: {fi_explainer_observational.importance_values}")
            print(f"sage_interventional: {sage_explainer_interventional.importance_values}")
            print(f"sage_observational: {sage_explainer_observational.importance_values}")
            print()

        if n % N_SAMPLES_STREAM == 0:
            break

    # Train model ----------------------------------------------------------------------------------
    for n, (x_i, y_i) in enumerate(iter_pandas(stream_1, y_1), start=1):

        y_pred = model.predict_one(x_i)
        training_metric.update(y_i, y_pred)

        model.learn_one(x_i, y_i)

        fi_explainer_interventional.explain_one(x_i, y_i)
        fi_explainer_observational.explain_one(x_i, y_i)

        sage_explainer_interventional.explain_one(x_i, y_i)
        sage_explainer_observational.explain_one(x_i, y_i)

        mdi_val, splits = mdi_explainer.explain_one()
        mdi_values.append(mdi_val)
        split_values.append(splits)

        fi_interventional_values.append(fi_explainer_interventional.importance_values)
        fi_observational_values.append(fi_explainer_observational.importance_values)
        sage_interventional_values.append(sage_explainer_interventional.importance_values)
        sage_observational_values.append(sage_explainer_observational.importance_values)

        if n % 1000 == 0:
            print(f"Trained on {n} samples")
            print(f"Training accuracy: {training_metric.get():.3f}")
            print(f"pfi_interventional: {fi_explainer_interventional.importance_values}")
            print(f"pfi_observational: {fi_explainer_observational.importance_values}")
            print(f"sage_interventional: {sage_explainer_interventional.importance_values}")
            print(f"sage_observational: {sage_explainer_observational.importance_values}")
            print()

        if n % N_SAMPLES_STREAM == 0:
            break

    fi_interventional_values = pd.DataFrame(fi_interventional_values, columns=feature_names)
    fi_observational_values = pd.DataFrame(fi_observational_values, columns=feature_names)
    sage_interventional_values = pd.DataFrame(sage_interventional_values, columns=feature_names)
    sage_observational_values = pd.DataFrame(sage_observational_values, columns=feature_names)
    mdi_values = pd.DataFrame(mdi_values, columns=feature_names)
    split_values = pd.DataFrame(split_values, columns=feature_names)

    # plotting -------------------------------------------------------------------------------------

    params = {
        'legend.fontsize': 'xx-large',
        'figure.figsize': (5, 5),
        'axes.labelsize': 'xx-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'xx-large',
        'ytick.labelsize': 'xx-large'
    }

    color_list = ['#44cfcb', '#44001A', '#4ea5d9', '#ef27a6', '#7d53de']

    plot_feature_names = [r"$X^1$", r"$X^2$", r"$X^3$", r"$X^4$"]
    type_explanation = ["interventional", "observational"]

    markevery = 50

    fig, axis = plt.subplots(1, 1, figsize=(6, 4))
    for i, feature_name in enumerate(feature_names):
        axis.plot(fi_interventional_values[feature_name][::markevery], color=color_list[i], ls="solid")
        axis.plot(fi_observational_values[feature_name][::markevery], color=color_list[i], ls="dashed")
    axis.set_title("gaussian stream", fontsize=16)
    axis.set_ylabel("PFI values", fontsize=14)
    axis.set_xlabel("Samples", fontsize=14)
    for i, feature_name in enumerate(plot_feature_names):
        axis.plot([], [], color=color_list[i], label=feature_name, ls="solid")
    axis.legend(title="$\\bf{Features}$")
    ax2 = axis.twinx()
    legend_line_int, = axis.plot([], [], color="black", label="interventional", ls="solid")
    legend_line_obs, = axis.plot([], [], color="black", label="observational", ls="dashed")
    legend_labels = ["interventional", "observational"]
    legend_lines = [legend_line_int, legend_line_obs]
    ax2.legend(legend_lines, legend_labels, edgecolor="0.8", fancybox=False, title="$\\bf{Removal}$", loc="center right")
    ax2.get_yaxis().set_visible(False)
    axis.axhline(y=0, color="gray", ls="dotted")
    plt.tight_layout()
    plt.savefig("gaussian_pfi.pdf")
    plt.show()

    fig, axis = plt.subplots(1, 1, figsize=(6, 4))
    for i, feature_name in enumerate(feature_names):
        axis.plot(sage_interventional_values[feature_name][::markevery], color=color_list[i], ls="solid")
        axis.plot(sage_observational_values[feature_name][::markevery], color=color_list[i], ls="dashed")
    axis.set_title("gaussian stream", fontsize=16)
    axis.set_ylabel("SAGE values", fontsize=14)
    axis.set_xlabel("Samples", fontsize=14)
    for i, feature_name in enumerate(plot_feature_names):
        axis.plot([], [], color=color_list[i], label=feature_name, ls="solid")
    axis.legend(title="$\\bf{Features}$")
    ax2 = axis.twinx()
    legend_line_int, = axis.plot([], [], color="black", label="interventional", ls="solid")
    legend_line_obs, = axis.plot([], [], color="black", label="observational", ls="dashed")
    legend_labels = ["interventional", "observational"]
    legend_lines = [legend_line_int, legend_line_obs]
    ax2.legend(legend_lines, legend_labels, edgecolor="0.8", fancybox=False, title="$\\bf{Removal}$", loc="center right")
    ax2.get_yaxis().set_visible(False)
    axis.axhline(y=0, color="gray", ls="dotted")
    plt.tight_layout()
    plt.savefig("sage_gaussian_stream.pdf")
    plt.show()

    fig, axis = plt.subplots(1, 1, figsize=(6, 4))
    for i, feature_name in enumerate(feature_names):
        axis.plot(mdi_values[feature_name][::markevery], color=color_list[i], ls="solid")
    axis.set_title("gaussian stream", fontsize=16)
    axis.set_ylabel("MDI values", fontsize=14)
    axis.set_xlabel("Samples", fontsize=14)
    for i, feature_name in enumerate(plot_feature_names):
        axis.plot([], [], color=color_list[i], label=feature_name, ls="solid")
    axis.legend(title="$\\bf{Features}$")
    axis.axhline(y=0, color="gray", ls="dotted")
    plt.tight_layout()
    plt.savefig("mdi_gaussian_stream.pdf")
    plt.show()

    plt.plot(split_values)
    plt.title('Splits')
    plt.show()

    # save results ---------------------------------------------------------------------------------
    fi_interventional_values.to_csv("fi_interventional_values.csv", index=False)
    fi_observational_values.to_csv("fi_observational_values.csv", index=False)
    sage_interventional_values.to_csv("sage_interventional_values.csv", index=False)
    sage_observational_values.to_csv("sage_observational_values.csv", index=False)
    mdi_values.to_csv("mdi_values.csv", index=False)
    split_values.to_csv("split_values.csv", index=False)
