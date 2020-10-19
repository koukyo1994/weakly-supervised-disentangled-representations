def anneal(init_val: float, max_val: float, step: int, anneal_steps: int):
    if anneal_steps == 0:
        return max_val

    assert max_val > init_val, "`max_val` must be greater than `init_val`"
    delta = max_val - init_val
    annealed = min((init_val + delta) * step / anneal_steps, max_val)
    return annealed
