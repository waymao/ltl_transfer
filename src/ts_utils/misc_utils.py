def generate_mean_rew_stop_fn(sliding_len, rew_threshold):
    """
    Generates a stop function that takes a running mean of the last `length` 
    rewards and returns True if the mean is better than `threshold`.
    """
    hist = []
    def stop_fn(mean_reward):
        nonlocal hist
        # average reward > threshold
        hist.append(mean_reward)

        if len(hist) > sliding_len:
            hist.pop(0)
        if len(hist) >= sliding_len and sum(hist) / len(hist) >= rew_threshold:
            return True
        else:
            return False
    return stop_fn
