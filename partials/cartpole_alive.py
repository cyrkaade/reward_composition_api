def partial_reward(obs, action, next_obs, true_reward, terminated, truncated, info):
    alive_bonus = 1.0 if not terminated else 0.0
    return {
        "partial": alive_bonus,
        "components": {"alive_bonus": alive_bonus},
    }

