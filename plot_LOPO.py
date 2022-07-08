import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def get_parser():
    """Set parameters for the experiment."""
    parser = argparse.ArgumentParser(
        "spike detection", description="spike detection using attention layer"
    )
    parser.add_argument("--path_data", type=str, default="../results/csv")
    parser.add_argument("--metric", type=str, default="f1")
    parser.add_argument("--n_subjects", type=int, default=1)
    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()  # you can modify this namespace for quick iterations
path_data = args.path_data
n_subjects = args.n_subjects
metric = args.metric

# Choose where to load the data
fnames = list(
    Path(path_data).glob("results_LOPO_spike_detection_{}-subjects.csv".format(n_subjects))
)

# concatene all the dataframe
df = pd.concat([pd.read_csv(fname) for fname in fnames], axis=0)

fig = plt.figure()
sns.boxplot(data=df, x="method", y=metric, palette="Set2")
sns.swarmplot(data=df, x="method", y=metric, hue="test_subject_id", palette="tab10")
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig.savefig(
            "../results/images/results_{}_score_{}_subjects.pdf".format(metric,
                                                                        n_subjects),
            bbox_inches="tight",
           )

# Create boxplot + swarmplot in a grid

# g = sns.FacetGrid(df.loc[(df['cost_sensitive'] is False)], row="mix_up", col="weight_loss", margin_titles=True)
# g.map(sns.swarmplot, "method", "f1", "test_subject_id", palette="tab10")
# g.map(sns.boxplot, "method", "f1", palette="Set2")
# g.add_legend()
# g.savefig(
#      "../results/images/results_LOPO_F1_score_{}_subjects.pdf".format(n_subjects),
#     bbox_inches="tight",
# )
# # g = sns.FacetGrid(df, row="mix_up", col="cost_sensitive", margin_titles=True)
# # g.map(sns.boxplot, "weight_loss", "f1", "method", palette="Set2") #, fit_reg=False, x_jitter=.1)
# # g.add_legend()
# # g.fig.suptitle('LOPO')
# # g.savefig(
# #      "../results/images/results_LOPO_F1_score_{}_subjects.pdf".format(n_subjects),
# #     bbox_inches="tight",
# # )

# print results
print(df.groupby(['method', 'mix_up', 'cost_sensitive', 'weight_loss', "transform"]).mean().reset_index())
print(df.groupby(['method', 'mix_up', 'cost_sensitive', 'weight_loss', "transform"]).std().reset_index())
