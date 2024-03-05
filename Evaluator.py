# Partially from: https://github.com/basiralab/DGL/blob/main/Project/evaluation_measures.py

from MatrixVectorizer import MatrixVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from numpy import linalg as LA
import torch
import numpy as np
import networkx as nx
from constants import *


def evaluate(
    truths_vectors,
    predictions_matrixes=None,
    predictions_vectors=None,
    include_diagonal=False,
    logs=False,
    include_fid=False,
):
    """
    Evaluate the performance of centrality prediction on graph data.

    Parameters:
    - truths_vectors (numpy array): Ground truth.
    - predictions_matrixes (numpy array, optional): Predicted adjacency matrices, if this parameter is not provided, predictions_vectors is required.
    - predictions_vectors (numpy array, optional): Predicted centrality values for nodes, if this parameter is not provided, predictions_matrixes is required.
    - include_diagonal (bool, optional): Include diagonal elements in computations.
    - logs (bool, optional): Print intermediate results if True.
    - include_fid (bool, optional): Include Frechet Inception Distance (FID) computation if True.

    Returns:
    - List containing [MAE, PCC, Jensen-Shannon Distance, Avg MAE Betweenness Centrality,
                      Avg MAE Eigenvector Centrality, Avg MAE PageRank Centrality].
                      If include_fid is True, FID is also included in the list.
    """

    # Check on optional inputs
    assert predictions_matrixes is not None or predictions_vectors is not None

    if predictions_matrixes is None:
        # Apply anti-vectorization
        predictions_matrixes = np.empty(
            (predictions_vectors.shape[0], HR_MATRIX_SIZE, HR_MATRIX_SIZE)
        )
        for i, prediction in enumerate(predictions_vectors):
            predictions_matrixes[i] = MatrixVectorizer.anti_vectorize(
                prediction, HR_MATRIX_SIZE, include_diagonal
            )
    else:
        # Apply vectorization
        predictions_vectors = np.empty((predictions_matrixes.shape[0], HR_ARRAY_SIZE))
        for i, prediction in enumerate(predictions_matrixes):
            predictions_vectors[i] = MatrixVectorizer.vectorize(
                prediction, include_diagonal
            )

    # Apply anti-vectorization on truth
    truths_matrixes = np.empty(
        (truths_vectors.shape[0], HR_MATRIX_SIZE, HR_MATRIX_SIZE)
    )
    for i, truth in enumerate(truths_vectors):
        truths_matrixes[i] = MatrixVectorizer.anti_vectorize(
            truth, HR_MATRIX_SIZE, include_diagonal
        )

    num_test_samples = predictions_matrixes.shape[0]

    # post-processing on predictions
    predictions_matrixes[predictions_matrixes < 0] = 0
    predictions_vectors[predictions_vectors < 0] = 0

    # Initialize lists to store MAEs for each centrality measure
    mae_bc = []
    mae_ec = []
    mae_pc = []
    pred_1d_list = []
    gt_1d_list = []

    # Iterate over each test sample
    for i in range(num_test_samples):

        if logs:
            print(i)

        # Convert adjacency matrices to NetworkX graphs
        pred_graph = nx.from_numpy_array(predictions_matrixes[i])
        gt_graph = nx.from_numpy_array(truths_matrixes[i])

        # Compute centrality measures
        pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
        pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight")
        pred_pc = nx.pagerank(pred_graph, weight="weight")

        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
        gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight")
        gt_pc = nx.pagerank(gt_graph, weight="weight")

        # Convert centrality dictionaries to lists
        pred_bc_values = list(pred_bc.values())
        pred_ec_values = list(pred_ec.values())
        pred_pc_values = list(pred_pc.values())

        gt_bc_values = list(gt_bc.values())
        gt_ec_values = list(gt_ec.values())
        gt_pc_values = list(gt_pc.values())

        # Compute MAEs
        mae_bc.append(mean_absolute_error(pred_bc_values, gt_bc_values))
        mae_ec.append(mean_absolute_error(pred_ec_values, gt_ec_values))
        mae_pc.append(mean_absolute_error(pred_pc_values, gt_pc_values))

        # Vectorize matrices
        pred_1d_list.append(MatrixVectorizer.vectorize(predictions_matrixes[i]))
        gt_1d_list.append(MatrixVectorizer.vectorize(truths_matrixes[i]))

    # Concatenate flattened matrices
    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    # Compute average MAEs
    avg_mae_bc = sum(mae_bc) / len(mae_bc)
    avg_mae_ec = sum(mae_ec) / len(mae_ec)
    avg_mae_pc = sum(mae_pc) / len(mae_pc)

    mae = mean_absolute_error(pred_1d, gt_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    js_dis = jensenshannon(pred_1d, gt_1d)

    if include_fid:
        # FID - Expensive computation
        mu_real = torch.mean(truths_vectors, dim=0)
        cov_real = torch.cov(truths_vectors.t())
        mu_gen = torch.mean(predictions_vectors, dim=0)
        cov_gen = torch.cov(predictions_vectors.t())
        fid = torch.sqrt(
            torch.norm(torch.sub(mu_real, mu_gen)) ** 2
            + torch.trace(
                cov_real + cov_gen - 2 * torch.sqrt(torch.matmul(cov_real, cov_gen))
            )
        )

    if logs:
        print("MAE: ", mae)
        print("PCC: ", pcc)
        print("Jensen-Shannon Distance: ", js_dis)
        print("Average MAE betweenness centrality:", avg_mae_bc)
        print("Average MAE eigenvector centrality:", avg_mae_ec)
        print("Average MAE PageRank centrality:", avg_mae_pc)
        if include_fid:
            print("Frechet Inception Distance:", fid)
    if include_fid:
        return [mae, pcc, js_dis, avg_mae_bc, avg_mae_ec, avg_mae_pc, fid]
    else:
        return [mae, pcc, js_dis, avg_mae_bc, avg_mae_ec, avg_mae_pc]
