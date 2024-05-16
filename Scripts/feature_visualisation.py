import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import polars as pl
import polars.selectors as cs


def main():
    """
    final_RP_df = pl.read_csv("../Data/healthy_rp.csv")
    final_RP_df_abnormal = pl.read_csv("../Data/unhealthy_rp.csv")
    healthy_googlenet_features_pretrained = (
        pl.read_csv("../Data/healthy_googlenet.csv").drop(cs.integer()).to_numpy()
    )
    unhealthy_googlenet_features_pretrained = (
        pl.read_csv("../Data/unhealthy_googlenet.csv").drop(cs.integer()).to_numpy()
    )

    # convert to PCA
    pca = PCA(n_components=2)

    X_healthy = np.nan_to_num(final_RP_df.fill_nan(0.0).to_numpy(), posinf=10.0)
    X_unhealthy = np.nan_to_num(
        final_RP_df_abnormal.fill_nan(0.0).to_numpy(), posinf=10.0
    )

    healthy = pca.fit_transform(X_healthy)
    healthy_var_ratio = pca.explained_variance_ratio_
    # healthy_covar = pca.get_covariance()

    unhealthy = pca.fit_transform(X_unhealthy)
    unhealthy_var_ratio = pca.explained_variance_ratio_
    # unhealthy_covar = pca.get_covariance()

    healthy_google_pretrained = pca.fit_transform(healthy_googlenet_features_pretrained)
    healthy_var_ratio_google_pretrained = pca.explained_variance_ratio_
    # healthy_covar_google_pretrained = pca.get_covariance()

    unhealthy_google_pretrained = pca.fit_transform(
        unhealthy_googlenet_features_pretrained
    )
    unhealthy_var_ratio_google_pretrained = pca.explained_variance_ratio_
    # unhealthy_covar_google_pretrained = pca.get_covariance()

    print(f"Sum of explained variance - Healthy_RQA: {sum(healthy_var_ratio)}")
    print(f"Sum of explained variance - Unhealthy_RQA: {sum(unhealthy_var_ratio)}")
    print(
        f"Sum of explained variance - Healthy_GoogLeNet: {sum(healthy_var_ratio_google_pretrained)}"
    )
    print(
        f"Sum of explained variance - Unhealthy_GoogLeNet: {sum(unhealthy_var_ratio_google_pretrained)}"
    )

    labels = [
        "RQA_Healthy",
        "RQA_Unhealthy",
        "GoogLeNet_Healthy",
        "GoogLeNet_Unhealthy",
    ]

    pca_result_df_healthy = pd.DataFrame(
        {"pca_1": healthy[:, 0], "pca_2": healthy[:, 1], "label": labels[0]}
    )
    pca_result_df_unhealthy = pd.DataFrame(
        {"pca_1": unhealthy[:, 0], "pca_2": unhealthy[:, 1], "label": labels[1]}
    )
    pca_result_df_googlenet_healthy = pd.DataFrame(
        {
            "pca_1": healthy_google_pretrained[:, 0],
            "pca_2": healthy_google_pretrained[:, 1],
            "label": labels[2],
        }
    )

    pca_result_df_googlenet_unhealthy = pd.DataFrame(
        {
            "pca_1": unhealthy_google_pretrained[:, 0],
            "pca_2": unhealthy_google_pretrained[:, 1],
            "label": labels[3],
        }
    )

    df = pd.concat(
        [
            pca_result_df_healthy,
            pca_result_df_unhealthy,
            pca_result_df_googlenet_healthy,
            pca_result_df_googlenet_unhealthy,
        ]
    )

    fig = px.scatter(df, x="pca_1", y="pca_2", color="label")
    fig.write_html("PCA_vis.html")
    """
    # import
    healthy_googlenet_features_pretrained = (
        pl.read_csv("../Data/healthy_googlenet.csv")
        .drop(cs.integer())
        .with_columns(target=pl.lit("healthy"))
    )

    unhealthy_googlenet_features_pretrained = (
        pl.read_csv("../Data/unhealthy_googlenet.csv")
        .drop(cs.integer())
        .with_columns(target=pl.lit("unhealthy"))
    )

    # save labels for plotting later
    healthy_labels = healthy_googlenet_features_pretrained.select(cs.string())
    unhealthy_labels = unhealthy_googlenet_features_pretrained.select(cs.string())

    # remove labels
    healthy_googlenet_features_pretrained = (
        healthy_googlenet_features_pretrained.select(~cs.string())
    )
    unhealthy_googlenet_features_pretrained = (
        unhealthy_googlenet_features_pretrained.select(~cs.string())
    )

    # concat df for tsne
    google_concat = pl.concat(
        [healthy_googlenet_features_pretrained, unhealthy_googlenet_features_pretrained]
    )

    # concat labels for plotting
    labels_concat = pl.concat([healthy_labels, unhealthy_labels]).to_series()

    # TSNE
    ts = TSNE(2, perplexity=50, n_iter=5000)
    embd = ts.fit_transform(google_concat)

    # generate plot
    result_df = pd.DataFrame(
        {"tsne_1": embd[:, 0], "tsne_2": embd[:, 1], "Label": labels_concat}
    )

    fig = px.scatter(result_df, x="tsne_1", y="tsne_2", color="Label")
    fig.write_html(f"NN.html")


if __name__ == "__main__":
    main()
