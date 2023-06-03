def zscore_avg_method(vertices):
    """
    Returns max(|v|) for visualizing zscores
    """
    val = 0
    for v in vertices:
        if abs(v) > abs(val):
            val = v
    return val