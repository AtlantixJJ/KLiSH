"""Define and test clustering metrics."""
from math import log
import torch
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
    fowlkes_mallows_score,
    rand_score,
)
from sklearn.metrics.cluster._supervised import (
    expected_mutual_information,
    _generalized_average,
    entropy,
)


__all__ = [
    "rand_index",
    "adjusted_rand_index",
    "fowlkes_mallows_index",
    "adjusted_mutual_info",
    "homogeneity_score",
]


def rand_index(n_samples, contingency):
    """Rand index.
    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings.
    The raw RI score is:
        RI = (number of agreeing pairs) / (number of pairs)
    Read more in the :ref:`User Guide <rand_score>`.
    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,), dtype=integral
        Cluster labels to evaluate.
    Returns
    -------
    RI : float
       Similarity score between 0.0 and 1.0, inclusive, 1.0 stands for
       perfect match.
    See Also
    --------
    adjusted_rand_score: Adjusted Rand Score
    adjusted_mutual_info_score: Adjusted Mutual Information
    Examples
    --------
    Perfectly matching labelings have a score of 1 even
      >>> from sklearn.metrics.cluster import rand_score
      >>> rand_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0
    Labelings that assign all classes members to the same clusters
    are complete but may not always be pure, hence penalized:
      >>> rand_score([0, 0, 1, 2], [0, 0, 1, 1])
      0.83...
    References
    ----------
    .. L. Hubert and P. Arabie, Comparing Partitions, Journal of
      Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075
    .. https://en.wikipedia.org/wiki/Simple_matching_coefficient
    .. https://en.wikipedia.org/wiki/Rand_index
    """
    c = pair_confusion_matrix(n_samples, contingency)
    numerator = c.diagonal().sum()
    denominator = c.sum()
    if numerator == denominator or denominator == 0:
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique
        # cluster. These are perfect matches hence return 1.0.
        return 1.0
    return numerator / denominator


def fowlkes_mallows_index(n_samples, contingency):
    """Measure the similarity of two clusterings of a set of points.
    .. versionadded:: 0.18
    The Fowlkes-Mallows index (FMI) is defined as the geometric mean between of
    the precision and recall::
        FMI = TP / sqrt((TP + FP) * (TP + FN))
    Where ``TP`` is the number of **True Positive** (i.e. the number of pair of
    points that belongs in the same clusters in both ``labels_true`` and
    ``labels_pred``), ``FP`` is the number of **False Positive** (i.e. the
    number of pair of points that belongs in the same clusters in
    ``labels_true`` and not in ``labels_pred``) and ``FN`` is the number of
    **False Negative** (i.e the number of pair of points that belongs in the
    same clusters in ``labels_pred`` and not in ``labels_True``).
    The score ranges from 0 to 1. A high value indicates a good similarity
    between two clusters.
    Read more in the :ref:`User Guide <fowlkes_mallows_scores>`.
    Parameters
    ----------
    labels_true : int array, shape = (``n_samples``,)
        A clustering of the data into disjoint subsets.
    labels_pred : array, shape = (``n_samples``, )
        A clustering of the data into disjoint subsets.
    sparse : bool, default=False
        Compute contingency matrix internally with sparse matrix.
    Returns
    -------
    score : float
       The resulting Fowlkes-Mallows score.
    Examples
    --------
    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::
      >>> from sklearn.metrics.cluster import fowlkes_mallows_score
      >>> fowlkes_mallows_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> fowlkes_mallows_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0
    If classes members are completely split across different clusters,
    the assignment is totally random, hence the FMI is null::
      >>> fowlkes_mallows_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0
    References
    ----------
    .. [1] `E. B. Fowkles and C. L. Mallows, 1983. "A method for comparing two
       hierarchical clusterings". Journal of the American Statistical
       Association
       <https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478008>`_
    .. [2] `Wikipedia entry for the Fowlkes-Mallows Index
           <https://en.wikipedia.org/wiki/Fowlkes-Mallows_index>`_
    """
    c = contingency
    tk = (contingency**2).sum() - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    return np.sqrt(tk / pk) * np.sqrt(tk / qk) if tk != 0.0 else 0.0


def mutual_info_score(contingency):
    """Mutual Information between two clusterings."""
    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    else:
        raise ValueError("Unsupported type for 'contingency': %s" % type(contingency))

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
        np.int64, copy=False
    )
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)


def entropy_count(label_count):
    """Calculates the entropy for a labeling.

    Parameters
    ----------
    labels : int array, shape = [n_samples]
        The labels

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    if len(label_count) == 0:
        return 1.0
    pi = label_count[label_count > 0].astype(np.float64)
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))


def adjusted_mutual_info(n_samples, contingency):
    """Adjusted Mutual Information between two clusterings."""
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(contingency)
    # Calculate the expected value for the mutual information
    emi = expected_mutual_information(contingency, n_samples)
    # Calculate entropy for each labeling
    h_true = entropy_count(contingency.sum(1))
    h_pred = entropy_count(contingency.sum(0))
    normalizer = _generalized_average(h_true, h_pred, "arithmetic")
    denominator = normalizer - emi
    # Avoid 0.0 / 0.0 when expectation equals maximum, i.e a perfect match.
    # normalizer should always be >= emi, but because of floating-point
    # representation, sometimes emi is slightly larger. Correct this
    # by preserving the sign.
    if denominator < 0:
        denominator = min(denominator, -np.finfo("float64").eps)
    else:
        denominator = max(denominator, np.finfo("float64").eps)
    ami = (mi - emi) / denominator
    return ami


def pair_confusion_matrix(n_samples, contingency):
    """Pair confusion matrix arising from two clusterings.
    The pair confusion matrix :math:`C` computes a 2 by 2 similarity matrix
    between two clusterings by considering all pairs of samples and counting
    pairs that are assigned into the same or into different clusters under
    the true and predicted clusterings.
    Considering a pair of samples that is clustered together a positive pair,
    then as in binary classification the count of true negatives is
    :math:`C_{00}`, false negatives is :math:`C_{10}`, true positives is
    :math:`C_{11}` and false positives is :math:`C_{01}`.
    Read more in the :ref:`User Guide <pair_confusion_matrix>`.
    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,), dtype=integral
        Cluster labels to evaluate.
    Returns
    -------
    C : ndarray of shape (2, 2), dtype=np.int64
        The contingency matrix.
    See Also
    --------
    rand_score: Rand Score
    adjusted_rand_score: Adjusted Rand Score
    adjusted_mutual_info_score: Adjusted Mutual Information
    Examples
    --------
    Perfectly matching labelings have all non-zero entries on the
    diagonal regardless of actual label values:
      >>> from sklearn.metrics.cluster import pair_confusion_matrix
      >>> pair_confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])
      array([[8, 0],
             [0, 4]]...
    Labelings that assign all classes members to the same clusters
    are complete but may be not always pure, hence penalized, and
    have some off-diagonal non-zero entries:
      >>> pair_confusion_matrix([0, 0, 1, 2], [0, 0, 1, 1])
      array([[8, 2],
             [0, 2]]...
    Note that the matrix is not symmetric.
    References
    ----------
    .. L. Hubert and P. Arabie, Comparing Partitions, Journal of
      Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075
    """
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency**2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples**2 - C[0, 1] - C[1, 0] - sum_squares
    return C


def adjusted_rand_index(n_samples, contingency):
    """Calculate ARI. Modified from sklearn."""
    (tn, fp), (fn, tp) = pair_confusion_matrix(n_samples, contingency)
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


def homogeneity_score(contingency, beta=1.0):
    """Compute the homogeneity and completeness and V-Measure scores at once.
    Those metrics are based on normalized conditional entropy measures of
    the clustering labeling to evaluate given the knowledge of a Ground
    Truth class labels of the same samples.
    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.
    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.
    Both scores have positive values between 0.0 and 1.0, larger values
    being desirable.
    Those 3 metrics are independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score values in any way.
    V-Measure is furthermore symmetric: swapping ``labels_true`` and
    ``label_pred`` will give the same score. This does not hold for
    homogeneity and completeness. V-Measure is identical to
    :func:`normalized_mutual_info_score` with the arithmetic averaging
    method.
    Read more in the :ref:`User Guide <homogeneity_completeness>`.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference
    labels_pred : array-like of shape (n_samples,)
        cluster labels to evaluate
    beta : float, default=1.0
        Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
        If ``beta`` is greater than 1, ``completeness`` is weighted more
        strongly in the calculation. If ``beta`` is less than 1,
        ``homogeneity`` is weighted more strongly.
    Returns
    -------
    homogeneity : float
       score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling
    completeness : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
    v_measure : float
        harmonic mean of the first two
    See Also
    --------
    homogeneity_score
    completeness_score
    v_measure_score
    """
    entropy_C = entropy_count(contingency.sum(1))
    entropy_K = entropy_count(contingency.sum(0))

    MI = mutual_info_score(contingency)

    homogeneity = MI / (entropy_C) if entropy_C else 1.0
    completeness = MI / (entropy_K) if entropy_K else 1.0

    if homogeneity + completeness == 0.0:
        v_measure_score = 0.0
    else:
        v_measure_score = (
            (1 + beta)
            * homogeneity
            * completeness
            / (beta * homogeneity + completeness)
        )

    return homogeneity, completeness, v_measure_score


def rel_diff(a, b):
    """Relative difference between a and b in percentage."""
    return abs((a - b) / b) * 100


if __name__ == "__main__":
    print("=> Running tests...")

    N = 10000
    C1 = 20
    C2 = 50
    label_true = np.random.randint(0, C1, size=(N,))
    label_dt = np.random.randint(0, C2, size=(N,))
    gt = torch.from_numpy(label_true)
    dt = torch.from_numpy(label_dt)

    gt_ri = rand_score(label_true, label_dt)
    gt_ari = adjusted_rand_score(label_true, label_dt)
    gt_ami = adjusted_mutual_info_score(label_true, label_dt)
    gt_fmi = fowlkes_mallows_score(label_true, label_dt)
    gt_hs, gt_cs, gt_vms = homogeneity_completeness_v_measure(label_true, label_dt)
    h_true, h_pred = entropy(label_true), entropy(label_dt)

    step_size = 10
    gt_bl = torch.zeros(step_size, C1)
    dt_bl = torch.zeros(step_size, C2)
    gt_count = torch.zeros(C1)
    dt_count = torch.zeros(C2)
    mat = 0
    for i in range(0, N, 10):
        st, ed = i, i + 10
        gt_bl.fill_(0).scatter_(1, gt[st:ed].unsqueeze(1), 1)
        dt_bl.fill_(0).scatter_(1, dt[st:ed].unsqueeze(1), 1)
        gt_count += gt_bl.sum(0)
        dt_count += dt_bl.sum(0)
        mat = mat + torch.matmul(gt_bl.permute(1, 0), dt_bl)
    mat = mat.numpy().astype("float64")
    gt_count, dt_count = gt_count.numpy(), dt_count.numpy()
    my_ri = rand_index(N, mat)
    my_ari = adjusted_rand_index(N, mat)
    my_ami = adjusted_mutual_info(N, mat)
    my_fmi = fowlkes_mallows_index(N, mat)
    my_hs, my_cs, my_vms = homogeneity_score(mat)
    my_h_true = entropy_count(gt_count)
    my_h_pred = entropy_count(dt_count)
    print(f"=> Entropy of label: {my_h_true - h_true}")
    print(f"=> Entropy of prediction: {my_h_pred - h_pred}")
    print(f"=> RI: {rel_diff(gt_ri, my_ri)}%")
    print(f"=> ARI: {rel_diff(gt_ari, my_ari)}%")
    print(f"=> AMI: {rel_diff(gt_ami, my_ami)}%")
    print(f"=> FMI: {rel_diff(gt_fmi, my_fmi)}%")
    print(f"=> Homogeneity: {rel_diff(gt_hs, my_hs)}%")
    print(f"=> Completeness: {rel_diff(gt_cs, my_cs)}%")
    print(f"=> V Measure Score: {rel_diff(gt_vms, my_vms)}%")
