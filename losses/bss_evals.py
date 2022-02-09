'''
This submodule collects useful functionality required across the task
submodules, such as preprocessing, validation, and common computations.
'''

import os
import inspect
import six

import numpy as np
# import numpy as np
import scipy.fftpack
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve
import collections
import itertools
import warnings


def index_labels(labels, case_sensitive=False):
    """Convert a list of string identifiers into numerical indices.

    Parameters
    ----------
    labels : list of strings, shape=(n,)
        A list of annotations, e.g., segment or chord labels from an
        annotation file.

    case_sensitive : bool
        Set to True to enable case-sensitive label indexing
        (Default value = False)

    Returns
    -------
    indices : list, shape=(n,)
        Numerical representation of ``labels``
    index_to_label : dict
        Mapping to convert numerical indices back to labels.
        ``labels[i] == index_to_label[indices[i]]``

    """

    label_to_index = {}
    index_to_label = {}

    # If we're not case-sensitive,
    if not case_sensitive:
        labels = [str(s).lower() for s in labels]

    # First, build the unique label mapping
    for index, s in enumerate(sorted(set(labels))):
        label_to_index[s] = index
        index_to_label[index] = s

    # Remap the labels to indices
    indices = [label_to_index[s] for s in labels]

    # Return the converted labels, and the inverse mapping
    return indices, index_to_label


def generate_labels(items, prefix='__'):
    """Given an array of items (e.g. events, intervals), create a synthetic label
    for each event of the form '(label prefix)(item number)'

    Parameters
    ----------
    items : list-like
        A list or array of events or intervals
    prefix : str
        This prefix will be prepended to all synthetically generated labels
        (Default value = '__')

    Returns
    -------
    labels : list of str
        Synthetically generated labels

    """
    return ['{}{}'.format(prefix, n) for n in range(len(items))]


def intervals_to_samples(intervals, labels, offset=0, sample_size=0.1,
                         fill_value=None):
    """Convert an array of labeled time intervals to annotated samples.

    Parameters
    ----------
    intervals : np.ndarray, shape=(n, d)
        An array of time intervals, as returned by
        :func:`mir_eval.io.load_intervals()` or
        :func:`mir_eval.io.load_labeled_intervals()`.
        The ``i`` th interval spans time ``intervals[i, 0]`` to
        ``intervals[i, 1]``.

    labels : list, shape=(n,)
        The annotation for each interval

    offset : float > 0
        Phase offset of the sampled time grid (in seconds)
        (Default value = 0)

    sample_size : float > 0
        duration of each sample to be generated (in seconds)
        (Default value = 0.1)

    fill_value : type(labels[0])
        Object to use for the label with out-of-range time points.
        (Default value = None)

    Returns
    -------
    sample_times : list
        list of sample times

    sample_labels : list
        array of labels for each generated sample

    Notes
    -----
        Intervals will be rounded down to the nearest multiple
        of ``sample_size``.

    """

    # Round intervals to the sample size
    num_samples = int(np.floor(intervals.max() / sample_size))
    sample_indices = np.arange(num_samples, dtype=np.float32)
    sample_times = (sample_indices*sample_size + offset).tolist()
    sampled_labels = interpolate_intervals(
        intervals, labels, sample_times, fill_value)

    return sample_times, sampled_labels


def interpolate_intervals(intervals, labels, time_points, fill_value=None):
    """Assign labels to a set of points in time given a set of intervals.

    Time points that do not lie within an interval are mapped to `fill_value`.

    Parameters
    ----------
    intervals : np.ndarray, shape=(n, 2)
        An array of time intervals, as returned by
        :func:`mir_eval.io.load_intervals()`.
        The ``i`` th interval spans time ``intervals[i, 0]`` to
        ``intervals[i, 1]``.

        Intervals are assumed to be disjoint.

    labels : list, shape=(n,)
        The annotation for each interval

    time_points : array_like, shape=(m,)
        Points in time to assign labels.  These must be in
        non-decreasing order.

    fill_value : type(labels[0])
        Object to use for the label with out-of-range time points.
        (Default value = None)

    Returns
    -------
    aligned_labels : list
        Labels corresponding to the given time points.

    Raises
    ------
    ValueError
        If `time_points` is not in non-decreasing order.
    """

    # Verify that time_points is sorted
    time_points = np.asarray(time_points)

    if np.any(time_points[1:] < time_points[:-1]):
        raise ValueError('time_points must be in non-decreasing order')

    aligned_labels = [fill_value] * len(time_points)

    starts = np.searchsorted(time_points, intervals[:, 0], side='left')
    ends = np.searchsorted(time_points, intervals[:, 1], side='right')

    for (start, end, lab) in zip(starts, ends, labels):
        aligned_labels[start:end] = [lab] * (end - start)

    return aligned_labels


def sort_labeled_intervals(intervals, labels=None):
    '''Sort intervals, and optionally, their corresponding labels
    according to start time.

    Parameters
    ----------
    intervals : np.ndarray, shape=(n, 2)
        The input intervals

    labels : list, optional
        Labels for each interval

    Returns
    -------
    intervals_sorted or (intervals_sorted, labels_sorted)
        Labels are only returned if provided as input
    '''

    idx = np.argsort(intervals[:, 0])

    intervals_sorted = intervals[idx]

    if labels is None:
        return intervals_sorted
    else:
        return intervals_sorted, [labels[_] for _ in idx]


def f_measure(precision, recall, beta=1.0):
    """Compute the f-measure from precision and recall scores.

    Parameters
    ----------
    precision : float in (0, 1]
        Precision
    recall : float in (0, 1]
        Recall
    beta : float > 0
        Weighting factor for f-measure
        (Default value = 1.0)

    Returns
    -------
    f_measure : float
        The weighted f-measure

    """

    if precision == 0 and recall == 0:
        return 0.0

    return (1 + beta**2)*precision*recall/((beta**2)*precision + recall)


def intervals_to_boundaries(intervals, q=5):
    """Convert interval times into boundaries.

    Parameters
    ----------
    intervals : np.ndarray, shape=(n_events, 2)
        Array of interval start and end-times
    q : int
        Number of decimals to round to. (Default value = 5)

    Returns
    -------
    boundaries : np.ndarray
        Interval boundary times, including the end of the final interval

    """

    return np.unique(np.ravel(np.round(intervals, decimals=q)))


def boundaries_to_intervals(boundaries):
    """Convert an array of event times into intervals

    Parameters
    ----------
    boundaries : list-like
        List-like of event times.  These are assumed to be unique
        timestamps in ascending order.

    Returns
    -------
    intervals : np.ndarray, shape=(n_intervals, 2)
        Start and end time for each interval
    """

    if not np.allclose(boundaries, np.unique(boundaries)):
        raise ValueError('Boundary times are not unique or not ascending.')

    intervals = np.asarray(list(zip(boundaries[:-1], boundaries[1:])))

    return intervals


def adjust_intervals(intervals,
                     labels=None,
                     t_min=0.0,
                     t_max=None,
                     start_label='__T_MIN',
                     end_label='__T_MAX'):
    """Adjust a list of time intervals to span the range ``[t_min, t_max]``.

    Any intervals lying completely outside the specified range will be removed.

    Any intervals lying partially outside the specified range will be cropped.

    If the specified range exceeds the span of the provided data in either
    direction, additional intervals will be appended.  If an interval is
    appended at the beginning, it will be given the label ``start_label``; if
    an interval is appended at the end, it will be given the label
    ``end_label``.

    Parameters
    ----------
    intervals : np.ndarray, shape=(n_events, 2)
        Array of interval start and end-times
    labels : list, len=n_events or None
        List of labels
        (Default value = None)
    t_min : float or None
        Minimum interval start time.
        (Default value = 0.0)
    t_max : float or None
        Maximum interval end time.
        (Default value = None)
    start_label : str or float or int
        Label to give any intervals appended at the beginning
        (Default value = '__T_MIN')
    end_label : str or float or int
        Label to give any intervals appended at the end
        (Default value = '__T_MAX')

    Returns
    -------
    new_intervals : np.ndarray
        Intervals spanning ``[t_min, t_max]``
    new_labels : list
        List of labels for ``new_labels``

    """

    # When supplied intervals are empty and t_max and t_min are supplied,
    # create one interval from t_min to t_max with the label start_label
    if t_min is not None and t_max is not None and intervals.size == 0:
        return np.array([[t_min, t_max]]), [start_label]
    # When intervals are empty and either t_min or t_max are not supplied,
    # we can't append new intervals
    elif (t_min is None or t_max is None) and intervals.size == 0:
        raise ValueError("Supplied intervals are empty, can't append new"
                         " intervals")

    if t_min is not None:
        # Find the intervals that end at or after t_min
        first_idx = np.argwhere(intervals[:, 1] >= t_min)

        if len(first_idx) > 0:
            # If we have events below t_min, crop them out
            if labels is not None:
                labels = labels[int(first_idx[0]):]
            # Clip to the range (t_min, +inf)
            intervals = intervals[int(first_idx[0]):]
        intervals = np.maximum(t_min, intervals)

        if intervals.min() > t_min:
            # Lowest boundary is higher than t_min:
            # add a new boundary and label
            intervals = np.vstack(([t_min, intervals.min()], intervals))
            if labels is not None:
                labels.insert(0, start_label)

    if t_max is not None:
        # Find the intervals that begin after t_max
        last_idx = np.argwhere(intervals[:, 0] > t_max)

        if len(last_idx) > 0:
            # We have boundaries above t_max.
            # Trim to only boundaries <= t_max
            if labels is not None:
                labels = labels[:int(last_idx[0])]
            # Clip to the range (-inf, t_max)
            intervals = intervals[:int(last_idx[0])]

        intervals = np.minimum(t_max, intervals)

        if intervals.max() < t_max:
            # Last boundary is below t_max: add a new boundary and label
            intervals = np.vstack((intervals, [intervals.max(), t_max]))
            if labels is not None:
                labels.append(end_label)

    return intervals, labels


def adjust_events(events, labels=None, t_min=0.0,
                  t_max=None, label_prefix='__'):
    """Adjust the given list of event times to span the range
    ``[t_min, t_max]``.

    Any event times outside of the specified range will be removed.

    If the times do not span ``[t_min, t_max]``, additional events will be
    added with the prefix ``label_prefix``.

    Parameters
    ----------
    events : np.ndarray
        Array of event times (seconds)
    labels : list or None
        List of labels
        (Default value = None)
    t_min : float or None
        Minimum valid event time.
        (Default value = 0.0)
    t_max : float or None
        Maximum valid event time.
        (Default value = None)
    label_prefix : str
        Prefix string to use for synthetic labels
        (Default value = '__')

    Returns
    -------
    new_times : np.ndarray
        Event times corrected to the given range.

    """
    if t_min is not None:
        first_idx = np.argwhere(events >= t_min)

        if len(first_idx) > 0:
            # We have events below t_min
            # Crop them out
            if labels is not None:
                labels = labels[int(first_idx[0]):]
            events = events[int(first_idx[0]):]

        if events[0] > t_min:
            # Lowest boundary is higher than t_min:
            # add a new boundary and label
            events = np.concatenate(([t_min], events))
            if labels is not None:
                labels.insert(0, '%sT_MIN' % label_prefix)

    if t_max is not None:
        last_idx = np.argwhere(events > t_max)

        if len(last_idx) > 0:
            # We have boundaries above t_max.
            # Trim to only boundaries <= t_max
            if labels is not None:
                labels = labels[:int(last_idx[0])]
            events = events[:int(last_idx[0])]

        if events[-1] < t_max:
            # Last boundary is below t_max: add a new boundary and label
            events = np.concatenate((events, [t_max]))
            if labels is not None:
                labels.append('%sT_MAX' % label_prefix)

    return events, labels


def intersect_files(flist1, flist2):
    """Return the intersection of two sets of filepaths, based on the file name
    (after the final '/') and ignoring the file extension.

    Examples
    --------
     >>> flist1 = ['/a/b/abc.lab', '/c/d/123.lab', '/e/f/xyz.lab']
     >>> flist2 = ['/g/h/xyz.npy', '/i/j/123.txt', '/k/l/456.lab']
     >>> sublist1, sublist2 = mir_eval.util.intersect_files(flist1, flist2)
     >>> print sublist1
     ['/e/f/xyz.lab', '/c/d/123.lab']
     >>> print sublist2
     ['/g/h/xyz.npy', '/i/j/123.txt']

    Parameters
    ----------
    flist1 : list
        first list of filepaths
    flist2 : list
        second list of filepaths

    Returns
    -------
    sublist1 : list
        subset of filepaths with matching stems from ``flist1``
    sublist2 : list
        corresponding filepaths from ``flist2``

    """
    def fname(abs_path):
        """Returns the filename given an absolute path.

        Parameters
        ----------
        abs_path :


        Returns
        -------

        """
        return os.path.splitext(os.path.split(abs_path)[-1])[0]

    fmap = dict([(fname(f), f) for f in flist1])
    pairs = [list(), list()]
    for f in flist2:
        if fname(f) in fmap:
            pairs[0].append(fmap[fname(f)])
            pairs[1].append(f)

    return pairs


def merge_labeled_intervals(x_intervals, x_labels, y_intervals, y_labels):
    r"""Merge the time intervals of two sequences.

    Parameters
    ----------
    x_intervals : np.ndarray
        Array of interval times (seconds)
    x_labels : list or None
        List of labels
    y_intervals : np.ndarray
        Array of interval times (seconds)
    y_labels : list or None
        List of labels

    Returns
    -------
    new_intervals : np.ndarray
        New interval times of the merged sequences.
    new_x_labels : list
        New labels for the sequence ``x``
    new_y_labels : list
        New labels for the sequence ``y``

    """
    align_check = [x_intervals[0, 0] == y_intervals[0, 0],
                   x_intervals[-1, 1] == y_intervals[-1, 1]]
    if False in align_check:
        raise ValueError(
            "Time intervals do not align; did you mean to call "
            "'adjust_intervals()' first?")
    time_boundaries = np.unique(
        np.concatenate([x_intervals, y_intervals], axis=0))
    output_intervals = np.array(
        [time_boundaries[:-1], time_boundaries[1:]]).T

    x_labels_out, y_labels_out = [], []
    x_label_range = np.arange(len(x_labels))
    y_label_range = np.arange(len(y_labels))
    for t0, _ in output_intervals:
        x_idx = x_label_range[(t0 >= x_intervals[:, 0])]
        x_labels_out.append(x_labels[x_idx[-1]])
        y_idx = y_label_range[(t0 >= y_intervals[:, 0])]
        y_labels_out.append(y_labels[y_idx[-1]])
    return output_intervals, x_labels_out, y_labels_out


def _bipartite_match(graph):
    """Find maximum cardinality matching of a bipartite graph (U,V,E).
    The input format is a dictionary mapping members of U to a list
    of their neighbors in V.

    The output is a dict M mapping members of V to their matches in U.

    Parameters
    ----------
    graph : dictionary : left-vertex -> list of right vertices
        The input bipartite graph.  Each edge need only be specified once.

    Returns
    -------
    matching : dictionary : right-vertex -> left vertex
        A maximal bipartite matching.

    """
    # Adapted from:
    #
    # Hopcroft-Karp bipartite max-cardinality matching and max independent set
    # David Eppstein, UC Irvine, 27 Apr 2002

    # initialize greedy matching (redundant, but faster than full search)
    matching = {}
    for u in graph:
        for v in graph[u]:
            if v not in matching:
                matching[v] = u
                break

    while True:
        # structure residual graph into layers
        # pred[u] gives the neighbor in the previous layer for u in U
        # preds[v] gives a list of neighbors in the previous layer for v in V
        # unmatched gives a list of unmatched vertices in final layer of V,
        # and is also used as a flag value for pred[u] when u is in the first
        # layer
        preds = {}
        unmatched = []
        pred = dict([(u, unmatched) for u in graph])
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)

        # repeatedly extend layering structure by another pair of layers
        while layer and not unmatched:
            new_layer = {}
            for u in layer:
                for v in graph[u]:
                    if v not in preds:
                        new_layer.setdefault(v, []).append(u)
            layer = []
            for v in new_layer:
                preds[v] = new_layer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        # did we finish layering without finding any alternating paths?
        if not unmatched:
            unlayered = {}
            for u in graph:
                for v in graph[u]:
                    if v not in preds:
                        unlayered[v] = None
            return matching

        def recurse(v):
            """Recursively search backward through layers to find alternating
            paths.  recursion returns true if found path, false otherwise
            """
            if v in preds:
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred:
                        pu = pred[u]
                        del pred[u]
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return True
            return False

        for v in unmatched:
            recurse(v)


def _outer_distance_mod_n(ref, est, modulus=12):
    """Compute the absolute outer distance modulo n.
    Using this distance, d(11, 0) = 1 (modulo 12)

    Parameters
    ----------
    ref : np.ndarray, shape=(n,)
        Array of reference values.
    est : np.ndarray, shape=(m,)
        Array of estimated values.
    modulus : int
        The modulus.
        12 by default for octave equivalence.

    Returns
    -------
    outer_distance : np.ndarray, shape=(n, m)
        The outer circular distance modulo n.

    """
    ref_mod_n = np.mod(ref, modulus)
    est_mod_n = np.mod(est, modulus)
    abs_diff = np.abs(np.subtract.outer(ref_mod_n, est_mod_n))
    return np.minimum(abs_diff, modulus - abs_diff)


def match_events(ref, est, window, distance=None):
    """Compute a maximum matching between reference and estimated event times,
    subject to a window constraint.

    Given two lists of event times ``ref`` and ``est``, we seek the largest set
    of correspondences ``(ref[i], est[j])`` such that
    ``distance(ref[i], est[j]) <= window``, and each
    ``ref[i]`` and ``est[j]`` is matched at most once.

    This is useful for computing precision/recall metrics in beat tracking,
    onset detection, and segmentation.

    Parameters
    ----------
    ref : np.ndarray, shape=(n,)
        Array of reference values
    est : np.ndarray, shape=(m,)
        Array of estimated values
    window : float > 0
        Size of the window.
    distance : function
        function that computes the outer distance of ref and est.
        By default uses ``|ref[i] - est[j]|``

    Returns
    -------
    matching : list of tuples
        A list of matched reference and event numbers.
        ``matching[i] == (i, j)`` where ``ref[i]`` matches ``est[j]``.

    """
    if distance is not None:
        # Compute the indices of feasible pairings
        hits = np.where(distance(ref, est) <= window)
    else:
        hits = _fast_hit_windows(ref, est, window)

    # Construct the graph input
    G = {}
    for ref_i, est_i in zip(*hits):
        if est_i not in G:
            G[est_i] = []
        G[est_i].append(ref_i)

    # Compute the maximum matching
    matching = sorted(_bipartite_match(G).items())

    return matching


def _fast_hit_windows(ref, est, window):
    '''Fast calculation of windowed hits for time events.

    Given two lists of event times ``ref`` and ``est``, and a
    tolerance window, computes a list of pairings
    ``(i, j)`` where ``|ref[i] - est[j]| <= window``.

    This is equivalent to, but more efficient than the following:

    >>> hit_ref, hit_est = np.where(np.abs(np.subtract.outer(ref, est))
    ...                             <= window)

    Parameters
    ----------
    ref : np.ndarray, shape=(n,)
        Array of reference values
    est : np.ndarray, shape=(m,)
        Array of estimated values
    window : float >= 0
        Size of the tolerance window

    Returns
    -------
    hit_ref : np.ndarray
    hit_est : np.ndarray
        indices such that ``|hit_ref[i] - hit_est[i]| <= window``
    '''

    ref = np.asarray(ref)
    est = np.asarray(est)
    ref_idx = np.argsort(ref)
    ref_sorted = ref[ref_idx]

    left_idx = np.searchsorted(ref_sorted, est - window, side='left')
    right_idx = np.searchsorted(ref_sorted, est + window, side='right')

    hit_ref, hit_est = [], []

    for j, (start, end) in enumerate(zip(left_idx, right_idx)):
        hit_ref.extend(ref_idx[start:end])
        hit_est.extend([j] * (end - start))

    return hit_ref, hit_est


def validate_intervals(intervals):
    """Checks that an (n, 2) interval ndarray is well-formed, and raises errors
    if not.

    Parameters
    ----------
    intervals : np.ndarray, shape=(n, 2)
        Array of interval start/end locations.

    """

    # Validate interval shape
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError('Intervals should be n-by-2 numpy ndarray, '
                         'but shape={}'.format(intervals.shape))

    # Make sure no times are negative
    if (intervals < 0).any():
        raise ValueError('Negative interval times found')

    # Make sure all intervals have strictly positive duration
    if (intervals[:, 1] <= intervals[:, 0]).any():
        raise ValueError('All interval durations must be strictly positive')


def validate_events(events, max_time=30000.):
    """Checks that a 1-d event location ndarray is well-formed, and raises
    errors if not.

    Parameters
    ----------
    events : np.ndarray, shape=(n,)
        Array of event times
    max_time : float
        If an event is found above this time, a ValueError will be raised.
        (Default value = 30000.)

    """
    # Make sure no event times are huge
    if (events > max_time).any():
        raise ValueError('An event at time {} was found which is greater than '
                         'the maximum allowable time of max_time = {} (did you'
                         ' supply event times in '
                         'seconds?)'.format(events.max(), max_time))
    # Make sure event locations are 1-d np ndarrays
    if events.ndim != 1:
        raise ValueError('Event times should be 1-d numpy ndarray, '
                         'but shape={}'.format(events.shape))
    # Make sure event times are increasing
    if (np.diff(events) < 0).any():
        raise ValueError('Events should be in increasing order.')


def validate_frequencies(frequencies, max_freq, min_freq,
                         allow_negatives=False):
    """Checks that a 1-d frequency ndarray is well-formed, and raises
    errors if not.

    Parameters
    ----------
    frequencies : np.ndarray, shape=(n,)
        Array of frequency values
    max_freq : float
        If a frequency is found above this pitch, a ValueError will be raised.
        (Default value = 5000.)
    min_freq : float
        If a frequency is found below this pitch, a ValueError will be raised.
        (Default value = 20.)
    allow_negatives : bool
        Whether or not to allow negative frequency values.
    """
    # If flag is true, map frequencies to their absolute value.
    if allow_negatives:
        frequencies = np.abs(frequencies)
    # Make sure no frequency values are huge
    if (np.abs(frequencies) > max_freq).any():
        raise ValueError('A frequency of {} was found which is greater than '
                         'the maximum allowable value of max_freq = {} (did '
                         'you supply frequency values in '
                         'Hz?)'.format(frequencies.max(), max_freq))
    # Make sure no frequency values are tiny
    if (np.abs(frequencies) < min_freq).any():
        raise ValueError('A frequency of {} was found which is less than the '
                         'minimum allowable value of min_freq = {} (did you '
                         'supply frequency values in '
                         'Hz?)'.format(frequencies.min(), min_freq))
    # Make sure frequency values are 1-d np ndarrays
    if frequencies.ndim != 1:
        raise ValueError('Frequencies should be 1-d numpy ndarray, '
                         'but shape={}'.format(frequencies.shape))


def has_kwargs(function):
    r'''Determine whether a function has \*\*kwargs.

    Parameters
    ----------
    function : callable
        The function to test

    Returns
    -------
    True if function accepts arbitrary keyword arguments.
    False otherwise.
    '''

    if six.PY2:
        return inspect.getargspec(function).keywords is not None
    else:
        sig = inspect.signature(function)

        for param in sig.parameters.values():
            if param.kind == param.VAR_KEYWORD:
                return True

        return False


def filter_kwargs(_function, *args, **kwargs):
    """Given a function and args and keyword args to pass to it, call the function
    but using only the keyword arguments which it accepts.  This is equivalent
    to redefining the function with an additional \*\*kwargs to accept slop
    keyword args.

    If the target function already accepts \*\*kwargs parameters, no filtering
    is performed.

    Parameters
    ----------
    _function : callable
        Function to call.  Can take in any number of args or kwargs

    """

    if has_kwargs(_function):
        return _function(*args, **kwargs)

    # Get the list of function arguments
    func_code = six.get_function_code(_function)
    function_args = func_code.co_varnames[:func_code.co_argcount]
    # Construct a dict of those kwargs which appear in the function
    filtered_kwargs = {}
    for kwarg, value in list(kwargs.items()):
        if kwarg in function_args:
            filtered_kwargs[kwarg] = value
    # Call the function with the supplied args and the filtered kwarg dict
    return _function(*args, **filtered_kwargs)


def intervals_to_durations(intervals):
    """Converts an array of n intervals to their n durations.

    Parameters
    ----------
    intervals : np.ndarray, shape=(n, 2)
        An array of time intervals, as returned by
        :func:`mir_eval.io.load_intervals()`.
        The ``i`` th interval spans time ``intervals[i, 0]`` to
        ``intervals[i, 1]``.

    Returns
    -------
    durations : np.ndarray, shape=(n,)
        Array of the duration of each interval.

    """
    validate_intervals(intervals)
    return np.abs(np.diff(intervals, axis=-1)).flatten()


def hz_to_midi(freqs):
    '''Convert Hz to MIDI numbers

    Parameters
    ----------
    freqs : number or ndarray
        Frequency/frequencies in Hz

    Returns
    -------
    midi : number or ndarray
        MIDI note numbers corresponding to input frequencies.
        Note that these may be fractional.
    '''
    return 12.0 * (np.log2(freqs) - np.log2(440.0)) + 69.0


def midi_to_hz(midi):
    '''Convert MIDI numbers to Hz

    Parameters
    ----------
    midi : number or ndarray
        MIDI notes

    Returns
    -------
    freqs : number or ndarray
        Frequency/frequencies in Hz corresponding to `midi`
    '''
    return 440.0 * (2.0 ** ((midi - 69.0)/12.0))

# -*- coding: utf-8 -*-
'''
Source separation algorithms attempt to extract recordings of individual
sources from a recording of a mixture of sources.  Evaluation methods for
source separation compare the extracted sources from reference sources and
attempt to measure the perceptual quality of the separation.

See also the bss_eval MATLAB toolbox:
    http://bass-db.gforge.inria.fr/bss_eval/

Conventions
-----------

An audio signal is expected to be in the format of a 1-dimensional array where
the entries are the samples of the audio signal.  When providing a group of
estimated or reference sources, they should be provided in a 2-dimensional
array, where the first dimension corresponds to the source number and the
second corresponds to the samples.

Metrics
-------

* :func:`mir_eval.separation.bss_eval_sources`: Computes the bss_eval_sources
  metrics from bss_eval, which optionally optimally match the estimated sources
  to the reference sources and measure the distortion and artifacts present in
  the estimated sources as well as the interference between them.

* :func:`mir_eval.separation.bss_eval_sources_framewise`: Computes the
  bss_eval_sources metrics on a frame-by-frame basis.

* :func:`mir_eval.separation.bss_eval_images`: Computes the bss_eval_images
  metrics from bss_eval, which includes the metrics in
  :func:`mir_eval.separation.bss_eval_sources` plus the image to spatial
  distortion ratio.

* :func:`mir_eval.separation.bss_eval_images_framewise`: Computes the
  bss_eval_images metrics on a frame-by-frame basis.

References
----------
  .. [#vincent2006performance] Emmanuel Vincent, Rémi Gribonval, and Cédric
      Févotte, "Performance measurement in blind audio source separation," IEEE
      Trans. on Audio, Speech and Language Processing, 14(4):1462-1469, 2006.


'''


# from . import util


# The maximum allowable number of sources (prevents insane computational load)
MAX_SOURCES = 100


def validate(reference_sources, estimated_sources):
    """Checks that the input data to a metric are valid, and throws helpful
    errors if not.

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources

    """

    if reference_sources.shape != estimated_sources.shape:
        raise ValueError('The shape of estimated sources and the true '
                         'sources should match.  reference_sources.shape '
                         '= {}, estimated_sources.shape '
                         '= {}'.format(reference_sources.shape,
                                       estimated_sources.shape))

    if reference_sources.ndim > 3 or estimated_sources.ndim > 3:
        raise ValueError('The number of dimensions is too high (must be less '
                         'than 3). reference_sources.ndim = {}, '
                         'estimated_sources.ndim '
                         '= {}'.format(reference_sources.ndim,
                                       estimated_sources.ndim))

    if reference_sources.size == 0:
        warnings.warn("reference_sources is empty, should be of size "
                      "(nsrc, nsample).  sdr, sir, sar, and perm will all "
                      "be empty np.ndarrays")
    elif _any_source_silent(reference_sources):
        raise ValueError('All the reference sources should be non-silent (not '
                         'all-zeros), but at least one of the reference '
                         'sources is all 0s, which introduces ambiguity to the'
                         ' evaluation. (Otherwise we can add infinitely many '
                         'all-zero sources.)')

    if estimated_sources.size == 0:
        warnings.warn("estimated_sources is empty, should be of size "
                      "(nsrc, nsample).  sdr, sir, sar, and perm will all "
                      "be empty np.ndarrays")
    elif _any_source_silent(estimated_sources):
        raise ValueError('All the estimated sources should be non-silent (not '
                         'all-zeros), but at least one of the estimated '
                         'sources is all 0s. Since we require each reference '
                         'source to be non-silent, having a silent estimated '
                         'source will result in an underdetermined system.')

    if (estimated_sources.shape[0] > MAX_SOURCES or
            reference_sources.shape[0] > MAX_SOURCES):
        raise ValueError('The supplied matrices should be of shape (nsrc,'
                         ' nsampl) but reference_sources.shape[0] = {} and '
                         'estimated_sources.shape[0] = {} which is greater '
                         'than mir_eval.separation.MAX_SOURCES = {}.  To '
                         'override this check, set '
                         'mir_eval.separation.MAX_SOURCES to a '
                         'larger value.'.format(reference_sources.shape[0],
                                                estimated_sources.shape[0],
                                                MAX_SOURCES))


def _any_source_silent(sources):
    """Returns true if the parameter sources has any silent first dimensions"""
    return np.any(np.all(np.sum(
        sources, axis=tuple(range(2, sources.ndim))) == 0, axis=1))


def bss_eval_sources(reference_sources, estimated_sources,
                     compute_permutation=True):
    """
    Ordering and measurement of the separation quality for estimated source
    signals in terms of filtered true source, interference and artifacts.

    The decomposition allows a time-invariant filter distortion of length
    512, as described in Section III.B of [#vincent2006performance]_.

    Passing ``False`` for ``compute_permutation`` will improve the computation
    performance of the evaluation; however, it is not always appropriate and
    is not the way that the BSS_EVAL Matlab toolbox computes bss_eval_sources.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, sir, sar,
    ...  perm) = mir_eval.separation.bss_eval_sources(reference_sources,
    ...                                               estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources (must have same shape as
        estimated_sources)
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources (must have same shape as
        reference_sources)
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations (True by default)

    Returns
    -------
    sdr : np.ndarray, shape=(nsrc,)
        vector of Signal to Distortion Ratios (SDR)
    sir : np.ndarray, shape=(nsrc,)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc,)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc,)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number ``perm[j]`` corresponds to
        true source number ``j``). Note: ``perm`` will be ``[0, 1, ...,
        nsrc-1]`` if ``compute_permutation`` is ``False``.

    References
    ----------
    .. [#] Emmanuel Vincent, Shoko Araki, Fabian J. Theis, Guido Nolte, Pau
        Bofill, Hiroshi Sawada, Alexey Ozerov, B. Vikrham Gowreesunker, Dominik
        Lutter and Ngoc Q.K. Duong, "The Signal Separation Evaluation Campaign
        (2007-2010): Achievements and remaining challenges", Signal Processing,
        92, pp. 1928-1936, 2012.

    """

    # make sure the input is of shape (nsrc, nsampl)
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if reference_sources.ndim == 1:
        reference_sources = reference_sources[np.newaxis, :]

    # validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    nsrc = estimated_sources.shape[0]

    # does user desire permutations?
    if compute_permutation:
        # compute criteria for all possible pair matches
        sdr = np.empty((nsrc, nsrc))
        sir = np.empty((nsrc, nsrc))
        sar = np.empty((nsrc, nsrc))
        for jest in range(nsrc):
            for jtrue in range(nsrc):
                s_true, e_spat, e_interf, e_artif = \
                    _bss_decomp_mtifilt(reference_sources,
                                        estimated_sources[jest],
                                        jtrue, 512)
                sdr[jest, jtrue], sir[jest, jtrue], sar[jest, jtrue] = \
                    _bss_source_crit(s_true, e_spat, e_interf, e_artif)

        # select the best ordering
        perms = list(itertools.permutations(list(range(nsrc))))
        mean_sir = np.empty(len(perms))
        dum = np.arange(nsrc)
        for (i, perm) in enumerate(perms):
            mean_sir[i] = np.mean(sir[perm, dum])
        popt = perms[np.argmax(mean_sir)]
        idx = (popt, dum)
        return (sdr[idx], sir[idx], sar[idx], np.asarray(popt))
    else:
        # compute criteria for only the simple correspondence
        # (estimate 1 is estimate corresponding to reference source 1, etc.)
        sdr = np.empty(nsrc)
        sir = np.empty(nsrc)
        sar = np.empty(nsrc)
        for j in range(nsrc):
            s_true, e_spat, e_interf, e_artif = \
                _bss_decomp_mtifilt(reference_sources,
                                    estimated_sources[j],
                                    j, 512)
            sdr[j], sir[j], sar[j] = \
                _bss_source_crit(s_true, e_spat, e_interf, e_artif)

        # return the default permutation for compatibility
        popt = np.arange(nsrc)
        return (sdr, sir, sar, popt)


def bss_eval_sources_framewise(reference_sources, estimated_sources,
                               window=30*44100, hop=15*44100,
                               compute_permutation=False):
    """Framewise computation of bss_eval_sources

    Please be aware that this function does not compute permutations (by
    default) on the possible relations between reference_sources and
    estimated_sources due to the dangers of a changing permutation. Therefore
    (by default), it assumes that ``reference_sources[i]`` corresponds to
    ``estimated_sources[i]``. To enable computing permutations please set
    ``compute_permutation`` to be ``True`` and check that the returned ``perm``
    is identical for all windows.

    NOTE: if ``reference_sources`` and ``estimated_sources`` would be evaluated
    using only a single window or are shorter than the window length, the
    result of :func:`mir_eval.separation.bss_eval_sources` called on
    ``reference_sources`` and ``estimated_sources`` (with the
    ``compute_permutation`` parameter passed to
    :func:`mir_eval.separation.bss_eval_sources`) is returned.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, sir, sar,
    ...  perm) = mir_eval.separation.bss_eval_sources_framewise(
             reference_sources,
    ...      estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources (must have the same shape as
        ``estimated_sources``)
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources (must have the same shape as
        ``reference_sources``)
    window : int, optional
        Window length for framewise evaluation (default value is 30s at a
        sample rate of 44.1kHz)
    hop : int, optional
        Hop size for framewise evaluation (default value is 15s at a
        sample rate of 44.1kHz)
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations for all windows
        (False by default)

    Returns
    -------
    sdr : np.ndarray, shape=(nsrc, nframes)
        vector of Signal to Distortion Ratios (SDR)
    sir : np.ndarray, shape=(nsrc, nframes)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc, nframes)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc, nframes)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number ``perm[j]`` corresponds to
        true source number ``j``).  Note: ``perm`` will be ``range(nsrc)`` for
        all windows if ``compute_permutation`` is ``False``

    """

    # make sure the input is of shape (nsrc, nsampl)
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if reference_sources.ndim == 1:
        reference_sources = reference_sources[np.newaxis, :]

    validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    nsrc = reference_sources.shape[0]

    nwin = int(
        np.floor((reference_sources.shape[1] - window + hop) / hop)
    )
    # if fewer than 2 windows would be evaluated, return the sources result
    if nwin < 2:
        result = bss_eval_sources(reference_sources,
                                  estimated_sources,
                                  compute_permutation)
        return [np.expand_dims(score, -1) for score in result]

    # compute the criteria across all windows
    sdr = np.empty((nsrc, nwin))
    sir = np.empty((nsrc, nwin))
    sar = np.empty((nsrc, nwin))
    perm = np.empty((nsrc, nwin))

    # k iterates across all the windows
    for k in range(nwin):
        win_slice = slice(k * hop, k * hop + window)
        ref_slice = reference_sources[:, win_slice]
        est_slice = estimated_sources[:, win_slice]
        # check for a silent frame
        if (not _any_source_silent(ref_slice) and
                not _any_source_silent(est_slice)):
            sdr[:, k], sir[:, k], sar[:, k], perm[:, k] = bss_eval_sources(
                ref_slice, est_slice, compute_permutation
            )
        else:
            # if we have a silent frame set results as np.nan
            sdr[:, k] = sir[:, k] = sar[:, k] = perm[:, k] = np.nan

    return sdr, sir, sar, perm


def bss_eval_images(reference_sources, estimated_sources,
                    compute_permutation=True):
    """Implementation of the bss_eval_images function from the
    BSS_EVAL Matlab toolbox.

    Ordering and measurement of the separation quality for estimated source
    signals in terms of filtered true source, interference and artifacts.
    This method also provides the ISR measure.

    The decomposition allows a time-invariant filter distortion of length
    512, as described in Section III.B of [#vincent2006performance]_.

    Passing ``False`` for ``compute_permutation`` will improve the computation
    performance of the evaluation; however, it is not always appropriate and
    is not the way that the BSS_EVAL Matlab toolbox computes bss_eval_images.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, isr, sir, sar,
    ...  perm) = mir_eval.separation.bss_eval_images(reference_sources,
    ...                                               estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl, nchan)
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl, nchan)
        matrix containing estimated sources
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations (True by default)

    Returns
    -------
    sdr : np.ndarray, shape=(nsrc,)
        vector of Signal to Distortion Ratios (SDR)
    isr : np.ndarray, shape=(nsrc,)
        vector of source Image to Spatial distortion Ratios (ISR)
    sir : np.ndarray, shape=(nsrc,)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc,)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc,)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number ``perm[j]`` corresponds to
        true source number ``j``).  Note: ``perm`` will be ``(1,2,...,nsrc)``
        if ``compute_permutation`` is ``False``.

    References
    ----------
    .. [#] Emmanuel Vincent, Shoko Araki, Fabian J. Theis, Guido Nolte, Pau
        Bofill, Hiroshi Sawada, Alexey Ozerov, B. Vikrham Gowreesunker, Dominik
        Lutter and Ngoc Q.K. Duong, "The Signal Separation Evaluation Campaign
        (2007-2010): Achievements and remaining challenges", Signal Processing,
        92, pp. 1928-1936, 2012.

    """

    # make sure the input has 3 dimensions
    # assuming input is in shape (nsampl) or (nsrc, nsampl)
    estimated_sources = np.atleast_3d(estimated_sources)
    reference_sources = np.atleast_3d(reference_sources)
    # we will ensure input doesn't have more than 3 dimensions in validate

    validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), \
                         np.array([]), np.array([])

    # determine size parameters
    nsrc = estimated_sources.shape[0]
    nsampl = estimated_sources.shape[1]
    nchan = estimated_sources.shape[2]

    # does the user desire permutation?
    if compute_permutation:
        # compute criteria for all possible pair matches
        sdr = np.empty((nsrc, nsrc))
        isr = np.empty((nsrc, nsrc))
        sir = np.empty((nsrc, nsrc))
        sar = np.empty((nsrc, nsrc))
        for jest in range(nsrc):
            for jtrue in range(nsrc):
                s_true, e_spat, e_interf, e_artif = \
                    _bss_decomp_mtifilt_images(
                        reference_sources,
                        np.reshape(
                            estimated_sources[jest],
                            (nsampl, nchan),
                            order='F'
                        ),
                        jtrue,
                        512
                    )
                sdr[jest, jtrue], isr[jest, jtrue], \
                    sir[jest, jtrue], sar[jest, jtrue] = \
                    _bss_image_crit(s_true, e_spat, e_interf, e_artif)

        # select the best ordering
        perms = list(itertools.permutations(range(nsrc)))
        mean_sir = np.empty(len(perms))
        dum = np.arange(nsrc)
        for (i, perm) in enumerate(perms):
            mean_sir[i] = np.mean(sir[perm, dum])
        popt = perms[np.argmax(mean_sir)]
        idx = (popt, dum)
        return (sdr[idx], isr[idx], sir[idx], sar[idx], np.asarray(popt))
    else:
        # compute criteria for only the simple correspondence
        # (estimate 1 is estimate corresponding to reference source 1, etc.)
        sdr = np.empty(nsrc)
        isr = np.empty(nsrc)
        sir = np.empty(nsrc)
        sar = np.empty(nsrc)
        Gj = [0] * nsrc  # prepare G matrics with zeroes
        G = np.zeros(1)
        for j in range(nsrc):
            # save G matrix to avoid recomputing it every call
            s_true, e_spat, e_interf, e_artif, Gj_temp, G = \
                _bss_decomp_mtifilt_images(reference_sources,
                                           np.reshape(estimated_sources[j],
                                                      (nsampl, nchan),
                                                      order='F'),
                                           j, 512, Gj[j], G)
            Gj[j] = Gj_temp
            sdr[j], isr[j], sir[j], sar[j] = \
                _bss_image_crit(s_true, e_spat, e_interf, e_artif)

        # return the default permutation for compatibility
        popt = np.arange(nsrc)
        return (sdr, isr, sir, sar, popt)


def bss_eval_images_framewise(reference_sources, estimated_sources,
                              window=30*44100, hop=15*44100,
                              compute_permutation=False):
    """Framewise computation of bss_eval_images

    Please be aware that this function does not compute permutations (by
    default) on the possible relations between ``reference_sources`` and
    ``estimated_sources`` due to the dangers of a changing permutation.
    Therefore (by default), it assumes that ``reference_sources[i]``
    corresponds to ``estimated_sources[i]``. To enable computing permutations
    please set ``compute_permutation`` to be ``True`` and check that the
    returned ``perm`` is identical for all windows.

    NOTE: if ``reference_sources`` and ``estimated_sources`` would be evaluated
    using only a single window or are shorter than the window length, the
    result of ``bss_eval_images`` called on ``reference_sources`` and
    ``estimated_sources`` (with the ``compute_permutation`` parameter passed to
    ``bss_eval_images``) is returned

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, isr, sir, sar,
    ...  perm) = mir_eval.separation.bss_eval_images_framewise(
             reference_sources,
    ...      estimated_sources,
             window,
    ....     hop)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl, nchan)
        matrix containing true sources (must have the same shape as
        ``estimated_sources``)
    estimated_sources : np.ndarray, shape=(nsrc, nsampl, nchan)
        matrix containing estimated sources (must have the same shape as
        ``reference_sources``)
    window : int
        Window length for framewise evaluation
    hop : int
        Hop size for framewise evaluation
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations for all windows
        (False by default)

    Returns
    -------
    sdr : np.ndarray, shape=(nsrc, nframes)
        vector of Signal to Distortion Ratios (SDR)
    isr : np.ndarray, shape=(nsrc, nframes)
        vector of source Image to Spatial distortion Ratios (ISR)
    sir : np.ndarray, shape=(nsrc, nframes)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc, nframes)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc, nframes)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number perm[j] corresponds to
        true source number j)
        Note: perm will be range(nsrc) for all windows if compute_permutation
        is False

    """

    # make sure the input has 3 dimensions
    # assuming input is in shape (nsampl) or (nsrc, nsampl)
    estimated_sources = np.atleast_3d(estimated_sources)
    reference_sources = np.atleast_3d(reference_sources)
    # we will ensure input doesn't have more than 3 dimensions in validate

    validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    nsrc = reference_sources.shape[0]

    nwin = int(
        np.floor((reference_sources.shape[1] - window + hop) / hop)
    )
    # if fewer than 2 windows would be evaluated, return the images result
    if nwin < 2:
        result = bss_eval_images(reference_sources,
                                 estimated_sources,
                                 compute_permutation)
        return [np.expand_dims(score, -1) for score in result]

    # compute the criteria across all windows
    sdr = np.empty((nsrc, nwin))
    isr = np.empty((nsrc, nwin))
    sir = np.empty((nsrc, nwin))
    sar = np.empty((nsrc, nwin))
    perm = np.empty((nsrc, nwin))

    # k iterates across all the windows
    for k in range(nwin):
        win_slice = slice(k * hop, k * hop + window)
        ref_slice = reference_sources[:, win_slice, :]
        est_slice = estimated_sources[:, win_slice, :]
        # check for a silent frame
        if (not _any_source_silent(ref_slice) and
                not _any_source_silent(est_slice)):
            sdr[:, k], isr[:, k], sir[:, k], sar[:, k], perm[:, k] = \
                bss_eval_images(
                    ref_slice, est_slice, compute_permutation
                )
        else:
            # if we have a silent frame set results as np.nan
            sdr[:, k] = sir[:, k] = sar[:, k] = perm[:, k] = np.nan

    return sdr, isr, sir, sar, perm


def _bss_decomp_mtifilt(reference_sources, estimated_source, j, flen):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    """
    nsampl = estimated_source.size
    # decomposition
    # true source image
    s_true = np.hstack((reference_sources[j], np.zeros(flen - 1)))
    # spatial (or filtering) distortion
    e_spat = _project(reference_sources[j, np.newaxis, :], estimated_source,
                      flen) - s_true
    # interference
    e_interf = _project(reference_sources,
                        estimated_source, flen) - s_true - e_spat
    # artifacts
    e_artif = -s_true - e_spat - e_interf
    e_artif[:nsampl] += estimated_source
    return (s_true, e_spat, e_interf, e_artif)


def _bss_decomp_mtifilt_images(reference_sources, estimated_source, j, flen,
                               Gj=None, G=None):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    Adapted version to work with multichannel sources.
    Improved performance can be gained by passing Gj and G parameters initially
    as all zeros. These parameters store the results from the computation of
    the G matrix in _project_images and then return them for subsequent calls
    to this function. This only works when not computing permuations.
    """
    nsampl = np.shape(estimated_source)[0]
    nchan = np.shape(estimated_source)[1]
    # are we saving the Gj and G parameters?
    saveg = Gj is not None and G is not None
    # decomposition
    # true source image
    s_true = np.hstack((np.reshape(reference_sources[j],
                                   (nsampl, nchan),
                                   order="F").transpose(),
                        np.zeros((nchan, flen - 1))))
    # spatial (or filtering) distortion
    if saveg:
        e_spat, Gj = _project_images(reference_sources[j, np.newaxis, :],
                                     estimated_source, flen, Gj)
    else:
        e_spat = _project_images(reference_sources[j, np.newaxis, :],
                                 estimated_source, flen)
    e_spat = e_spat - s_true
    # interference
    if saveg:
        e_interf, G = _project_images(reference_sources,
                                      estimated_source, flen, G)
    else:
        e_interf = _project_images(reference_sources,
                                   estimated_source, flen)
    e_interf = e_interf - s_true - e_spat
    # artifacts
    e_artif = -s_true - e_spat - e_interf
    e_artif[:, :nsampl] += estimated_source.transpose()
    # return Gj and G only if they were passed in
    if saveg:
        return (s_true, e_spat, e_interf, e_artif, Gj, G)
    else:
        return (s_true, e_spat, e_interf, e_artif)


def _project(reference_sources, estimated_source, flen):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    """
    nsrc = reference_sources.shape[0]
    nsampl = reference_sources.shape[1]

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = np.hstack((reference_sources,
                                   np.zeros((nsrc, flen - 1))))
    estimated_source = np.hstack((estimated_source, np.zeros(flen - 1)))
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
    sf = scipy.fftpack.fft(reference_sources, n=n_fft, axis=1)
    sef = scipy.fftpack.fft(estimated_source, n=n_fft)
    # inner products between delayed versions of reference_sources
    G = np.zeros((nsrc * flen, nsrc * flen))
    for i in range(nsrc):
        for j in range(nsrc):
            ssf = sf[i] * np.conj(sf[j])
            ssf = np.real(scipy.fftpack.ifft(ssf))
            ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                          r=ssf[:flen])
            G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
            G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T
    # inner products between estimated_source and delayed versions of
    # reference_sources
    D = np.zeros(nsrc * flen)
    for i in range(nsrc):
        ssef = sf[i] * np.conj(sef)
        ssef = np.real(scipy.fftpack.ifft(ssef))
        D[i * flen: (i+1) * flen] = np.hstack((ssef[0], ssef[-1:-flen:-1]))

    # Computing projection
    # Distortion filters
    try:
        C = np.linalg.solve(G, D).reshape(flen, nsrc, order='F')
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(flen, nsrc, order='F')
    # Filtering
    sproj = np.zeros(nsampl + flen - 1)
    for i in range(nsrc):
        sproj += fftconvolve(C[:, i], reference_sources[i])[:nsampl + flen - 1]
    return sproj


def _project_images(reference_sources, estimated_source, flen, G=None):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1.
    Passing G as all zeros will populate the G matrix and return it so it can
    be passed into the next call to avoid recomputing G (this will only works
    if not computing permutations).
    """
    nsrc = reference_sources.shape[0]
    nsampl = reference_sources.shape[1]
    nchan = reference_sources.shape[2]
    reference_sources = np.reshape(np.transpose(reference_sources, (2, 0, 1)),
                                   (nchan*nsrc, nsampl), order='F')

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = np.hstack((reference_sources,
                                   np.zeros((nchan*nsrc, flen - 1))))
    estimated_source = \
        np.hstack((estimated_source.transpose(), np.zeros((nchan, flen - 1))))
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
    sf = scipy.fftpack.fft(reference_sources, n=n_fft, axis=1)
    sef = scipy.fftpack.fft(estimated_source, n=n_fft)

    # inner products between delayed versions of reference_sources
    if G is None:
        saveg = False
        G = np.zeros((nchan * nsrc * flen, nchan * nsrc * flen))
        for i in range(nchan * nsrc):
            for j in range(i+1):
                ssf = sf[i] * np.conj(sf[j])
                ssf = np.real(scipy.fftpack.ifft(ssf))
                ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                              r=ssf[:flen])
                G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
                G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T
    else:  # avoid recomputing G (only works if no permutation is desired)
        saveg = True  # return G
        if np.all(G == 0):  # only compute G if passed as 0
            G = np.zeros((nchan * nsrc * flen, nchan * nsrc * flen))
            for i in range(nchan * nsrc):
                for j in range(i+1):
                    ssf = sf[i] * np.conj(sf[j])
                    ssf = np.real(scipy.fftpack.ifft(ssf))
                    ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                                  r=ssf[:flen])
                    G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
                    G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T

    # inner products between estimated_source and delayed versions of
    # reference_sources
    D = np.zeros((nchan * nsrc * flen, nchan))
    for k in range(nchan * nsrc):
        for i in range(nchan):
            ssef = sf[k] * np.conj(sef[i])
            ssef = np.real(scipy.fftpack.ifft(ssef))
            D[k * flen: (k+1) * flen, i] = \
                np.hstack((ssef[0], ssef[-1:-flen:-1])).transpose()

    # Computing projection
    # Distortion filters
    try:
        C = np.linalg.solve(G, D).reshape(flen, nchan*nsrc, nchan, order='F')
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(flen, nchan*nsrc, nchan,
                                             order='F')
    # Filtering
    sproj = np.zeros((nchan, nsampl + flen - 1))
    for k in range(nchan * nsrc):
        for i in range(nchan):
            sproj[i] += fftconvolve(C[:, k, i].transpose(),
                                    reference_sources[k])[:nsampl + flen - 1]
    # return G only if it was passed in
    if saveg:
        return sproj, G
    else:
        return sproj


def _bss_source_crit(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.
    """
    # energy ratios
    s_filt = s_true + e_spat
    sdr = _safe_db(np.sum(s_filt**2), np.sum((e_interf + e_artif)**2))
    sir = _safe_db(np.sum(s_filt**2), np.sum(e_interf**2))
    sar = _safe_db(np.sum((s_filt + e_interf)**2), np.sum(e_artif**2))
    return (sdr, sir, sar)


def _bss_image_crit(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given image in terms of
    filtered true source, spatial error, interference and artifacts.
    """
    # energy ratios
    sdr = _safe_db(np.sum(s_true**2), np.sum((e_spat+e_interf+e_artif)**2))
    isr = _safe_db(np.sum(s_true**2), np.sum(e_spat**2))
    sir = _safe_db(np.sum((s_true+e_spat)**2), np.sum(e_interf**2))
    sar = _safe_db(np.sum((s_true+e_spat+e_interf)**2), np.sum(e_artif**2))
    return (sdr, isr, sir, sar)


def _safe_db(num, den):
    """Properly handle the potential +Inf db SIR, instead of raising a
    RuntimeWarning. Only denominator is checked because the numerator can never
    be 0.
    """
    if den == 0:
        return np.Inf
    return 10 * np.log10(num / den)


def evaluate(reference_sources, estimated_sources, **kwargs):
    """Compute all metrics for the given reference and estimated signals.

    NOTE: This will always compute :func:`mir_eval.separation.bss_eval_images`
    for any valid input and will additionally compute
    :func:`mir_eval.separation.bss_eval_sources` for valid input with fewer
    than 3 dimensions.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated source
    >>> scores = mir_eval.separation.evaluate(reference_sources,
    ...                                       estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl[, nchan])
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl[, nchan])
        matrix containing estimated sources
    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.

    Returns
    -------
    scores : dict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.

    """
    # Compute all the metrics
    scores = collections.OrderedDict()

    sdr, isr, sir, sar, perm = util.filter_kwargs(
        bss_eval_images,
        reference_sources,
        estimated_sources,
        **kwargs
    )
    scores['Images - Source to Distortion'] = sdr.tolist()
    scores['Images - Image to Spatial'] = isr.tolist()
    scores['Images - Source to Interference'] = sir.tolist()
    scores['Images - Source to Artifact'] = sar.tolist()
    scores['Images - Source permutation'] = perm.tolist()

    sdr, isr, sir, sar, perm = util.filter_kwargs(
        bss_eval_images_framewise,
        reference_sources,
        estimated_sources,
        **kwargs
    )
    scores['Images Frames - Source to Distortion'] = sdr.tolist()
    scores['Images Frames - Image to Spatial'] = isr.tolist()
    scores['Images Frames - Source to Interference'] = sir.tolist()
    scores['Images Frames - Source to Artifact'] = sar.tolist()
    scores['Images Frames - Source permutation'] = perm.tolist()

    # Verify we can compute sources on this input
    if reference_sources.ndim < 3 and estimated_sources.ndim < 3:
        sdr, sir, sar, perm = util.filter_kwargs(
            bss_eval_sources_framewise,
            reference_sources,
            estimated_sources,
            **kwargs
        )
        scores['Sources Frames - Source to Distortion'] = sdr.tolist()
        scores['Sources Frames - Source to Interference'] = sir.tolist()
        scores['Sources Frames - Source to Artifact'] = sar.tolist()
        scores['Sources Frames - Source permutation'] = perm.tolist()

        sdr, sir, sar, perm = util.filter_kwargs(
            bss_eval_sources,
            reference_sources,
            estimated_sources,
            **kwargs
        )
        scores['Sources - Source to Distortion'] = sdr.tolist()
        scores['Sources - Source to Interference'] = sir.tolist()
        scores['Sources - Source to Artifact'] = sar.tolist()
        scores['Sources - Source permutation'] = perm.tolist()

    return scores


# import tensorflow as tf
# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5)
#
# model.evaluate(x_test,  y_test, verbose=2)