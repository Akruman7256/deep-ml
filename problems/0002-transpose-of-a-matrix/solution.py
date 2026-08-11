def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here

    n_rows = len(a)
    n_columns = len(a[0])

    final = []

    for i in range(n_columns):
        b = []
        for j in range(n_rows):
            b.append(a[j][i])
        final.append(b)
    return final

    pass