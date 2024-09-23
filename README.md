A comment on data formatting:
- data arrays should take the form $[x(t_1), x(t_2), ..., x(t_n)]$
- thus, arrays should have the form (rows, cols) == (n_state_vars, n_samples)
    - so, x[0] should be many samples of the first state variable
    - note the syntax "np.array([row1, row2, ...])"


A comment on state variable naming/displaying:
- we will name state varibles with 0 indexing. That is, $x_0$ ... $x_{n-1}$
