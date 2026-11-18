import numpy as np


class LeaveKOut:
    def __init__(self, k = 1, shuffle = False, random_state = 1):
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state

    def __call__(self, groups):
        group_ids = np.unique(groups)
        if not 1 <= self.k < len(group_ids):
            raise ValueError("leave_k_out: k parameter not valid")
        if self.shuffle:
            group_ids = np.random.default_rng(self.random_state).permutation(group_ids)
        for i in range(0, len(group_ids), self.k):
            held = group_ids[i: i + self.k]
            mask = np.isin(groups, held)
            yield np.flatnonzero(~mask), np.flatnonzero(mask)
