import pandas as pd
import river.metrics
from matplotlib import pyplot as plt
from river.utils import Rolling
from river.ensemble import BaggingClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.metrics import Accuracy

from ixai.explainer import IncrementalPFI
from ixai.storage import TreeStorage
from ixai.imputer import TreeImputer
from experiments.data.stream.synth import Agrawal

if __name__ == "__main__":

    # Get Data -------------------------------------------------------------------------------------
    N_SAMPLES = 20_000

    RANDOM_SEED: int = 42

    class_fn = 4

    stream_1 = Agrawal(classification_function=class_fn, random_seed=RANDOM_SEED, balance_classes=True)
    feature_names = stream_1.feature_names
    cat_feature_names = stream_1.cat_feature_names
    num_feature_names = stream_1.num_feature_names
    #feature_names = list([x_0 for x_0, _ in stream_1.take(1)][0].keys())

    loss_metric = river.metrics.Accuracy()
    training_metric = Rolling(river.metrics.Accuracy(), window_size=1000)

    model = BaggingClassifier(
        model=HoeffdingAdaptiveTreeClassifier(RANDOM_SEED),
        n_models=10,
        seed=RANDOM_SEED
    )
    # Get imputer and explainers -------------------------------------------------------------------
    model_function = model.predict_one

    # explainer interventional ---------------------------------------------------------------------
    smoothing_alpha = 0.001
    fi_explainer_interventional = IncrementalPFI(
        model_function=model_function,
        loss_function=Accuracy(),
        feature_names=feature_names,
        n_inner_samples=5,
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
        direct_predict_numeric=False
    )

    fi_explainer_observational = IncrementalPFI(
        model_function=model_function,
        loss_function=Accuracy(),
        feature_names=feature_names,
        n_inner_samples=5,
        smoothing_alpha=smoothing_alpha,
        storage=storage_observational,
        imputer=imputer_observational
    )

    fi_interventional_values = []
    fi_observational_values = []

    for (t, (x_i, y_i)) in enumerate(stream_1, start=1):
        # inference
        y_i_pred = model.predict_one(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)

        # explanation interventional
        pfi_interventional = fi_explainer_interventional.explain_one(x_i, y_i)
        fi_interventional_values.append(pfi_interventional)

        # explanation observational
        pfi_observational = fi_explainer_observational.explain_one(x_i, y_i)
        fi_observational_values.append(pfi_observational)

        # learning
        model.learn_one(x_i, y_i)

        if t % 1000 == 0:
            print(f"{t}: {training_metric.get():.3f}\n"
                  f"{t}: {pfi_interventional}\n"
                  f"{t}: {pfi_observational}\n")

        if t >= N_SAMPLES:
            break

    fi_interventional_values = pd.DataFrame(fi_interventional_values)
    fi_observational_values = pd.DataFrame(fi_observational_values)

    # Plot -----------------------------------------------------------------------------------------
    markevery = 100

    color_list = ['#44cfcb', '#44001A', '#4ea5d9', '#ef27a6', '#7d53de']

    fig, axis = plt.subplots(1, 1, figsize=(6, 4))

    axis.plot(fi_interventional_values["salary"][::markevery], label=r"$X^{(salary)}$ (interventional)", color=color_list[0], ls="solid", linewidth=2)
    axis.plot(fi_observational_values["salary"][::markevery], label=r"$X^{(salary)}$ (observational)", color=color_list[0], ls="dashed", linewidth=2)
    axis.plot(fi_interventional_values["commission"][::markevery], label=r"$X^{(commission)}$ (interventional)", color=color_list[1], ls="solid", linewidth=2)
    axis.plot(fi_observational_values["commission"][::markevery], label=r"$X^{(commission)}$ (observational)", color=color_list[1], ls="dashed", linewidth=2)
    axis.set_title(f"agrawal stream (concept {class_fn + 1})", fontsize=16)
    axis.set_xlabel("Samples", fontsize=14)
    axis.set_ylabel("PFI values", fontsize=14)
    axis.legend(ncols=2, loc="upper center")
    axis.axhline(y=0, color="gray", ls="dotted")

    if class_fn == 4:
        axis.set_ylim(-0.02, 0.19)
    if class_fn == 3:
        axis.set_ylim(-0.02, 0.54)

    plt.tight_layout()
    plt.savefig(f"agrawal_{class_fn + 1}.pdf")
    plt.show()

    # save results ---------------------------------------------------------------------------------
    fi_interventional_values.to_csv(f"agrawal_{class_fn + 1}_interventional.csv", index=False)
    fi_observational_values.to_csv(f"agrawal_{class_fn + 1}_observational.csv", index=False)
