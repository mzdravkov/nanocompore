import argparse

from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch

from gmm_gpu.gmm import GMM
from matplotlib.figure import Figure, SubFigure

from nanocompore.api import get_pos
from nanocompore.config import Config


def plot_gmm(args: argparse.Namespace, config: Config) -> Union[Figure, SubFigure, None]:
    df = get_pos(config, args.reference, int(args.position))
    print(df)
    measurements = df.loc[:, ['intensity', 'dwell']].to_numpy()
    tensor = torch.tensor(measurements).unsqueeze(0)
    gmm = GMM(n_components=2,
              device='cpu',
        random_seed=42,
              dtype=torch.float32)
    gmm.fit(tensor)

    # Means is a list with the means for each component.
    # The shape of each is (Points, Dims). We have a single point.
    c1_mean = gmm.means[0][0]
    c2_mean = gmm.means[1][0]
    # Covs is a list with the cov matrices for the components.
    # The shape of each is (Points, Dims, Dims).
    c1_cov = gmm.covs[0][0]
    c2_cov = gmm.covs[1][0]

    x1, y1 = np.random.multivariate_normal(c1_mean, c1_cov, 1000).T
    x2, y2 = np.random.multivariate_normal(c2_mean, c2_cov, 1000).T
    sampled_gaussians = pd.DataFrames(
        {'x': np.concatenate([x1, x2]),
         'y': np.concatenate([y1, y2]),
         'cluster': np.concatenate([np.full((1000,), 0),
                                    np.full((1000,), 1)])})
    sns.kdeplot(sampled_gaussians, x='x', y='y', levels=5, hue="cluster")
    ax = sns.scatterplot(df, x='dwell', y='intensity', hue='condition')
    return ax.get_figure()
