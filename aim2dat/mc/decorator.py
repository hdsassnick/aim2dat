

def mc_move(n_rand_nr):
    """Decorate functions that represetn Monte Carlo moves."""

    def decorator(func):
        func._n_rand_nrs = n_rand_nr

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator

