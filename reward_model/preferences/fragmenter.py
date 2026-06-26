from local_gym.wrappers.buffering_wrapper import Trajectory
import random
from reward_model.reward_model import RewardModel, preference_prob
from copy import deepcopy
import torch as th


def make_convert_trajectory(continuous):
    if continuous:
        return lambda traj: [[*state['obs'], *state['act'], state['partial_rew']] for state in traj.states]
    return lambda traj: [[*state['obs'], state['act'], state['partial_rew']] for state in traj.states]

class Fragmenter:
    def __init__(self, continuous, fragment_length=25, oversampling_rate=2.0):
        self.fragment_length = fragment_length
        self.fragments = []
        self.convert_traj = make_convert_trajectory(continuous)
        self.oversampling_rate = oversampling_rate if oversampling_rate >= 1.0 else 1.0

    def fragment_trajectories(self, trajectories: list[Trajectory]):
        for trajectory in trajectories:
            t_states = trajectory.get_states()
            t_fragments = [[t_states[i] for i in range(start_idx, start_idx+self.fragment_length)] for start_idx in range(0, len(t_states), self.fragment_length) if start_idx+self.fragment_length < len(t_states)]
            for t_fragment in t_fragments:
                self.fragments.append(Trajectory(t_fragment))
        return self.fragments

    def get_fragments(self):
        return self.fragments
    
    def shuffle_and_pair(self, array):
        random.shuffle(array)
        pairs = list(zip(array[::2], array[1::2]))
        return pairs
    
    def _down_sample_pairs(self, pairs):
        return pairs[:int(len(pairs)/self.oversampling_rate)]
    
    def get_random_pairs(self):
        shuffled_fragments = self.fragments
        pairs = self.shuffle_and_pair(shuffled_fragments)
        return self._down_sample_pairs(pairs)
    
    def dropout_active_learning(self, model: RewardModel, k: int = 8, dropout_p = 0.25, delta=False, n_batches=2048):
        mc_dropout_models = [deepcopy(model).dropout(dropout_p) for _ in range(k)]
        possible_pair_indexes = list(range(len(self.fragments)))

        fragments_list = [self.convert_traj(f) for f in self.fragments]
        fragments_tensor = th.Tensor(fragments_list)

        pred_rewards = [th.sum(mc_dropout_models[i].forward(fragments_tensor), dim=[1, 2]).tolist() for i in range(k)]
        fn_rewards = [sum([state['partial_rew'] for state in trajectory.states]) for trajectory in self.fragments]
        if delta:
            pred_rewards = [[pr + fr for pr, fr in zip(pred_rewards[j], fn_rewards)] for j in range(k)]

        indices_batches = [self.shuffle_and_pair(possible_pair_indexes) for _ in range(n_batches)]
        max_var = 0
        best_indices_batch = None
        for indices_batch in indices_batches:
            pair_probs = th.Tensor([[(pred_rewards[j][pi[0]], pred_rewards[j][pi[1]]) for pi in indices_batch] for j in range(k)])
            pred_preferences = preference_prob(pair_probs, 2)
            vars = th.sum(th.var(pred_preferences, dim=0), dim=1).tolist()
            
            if sum(vars) > max_var:
                max_var = sum(vars)
                best_indices_batch = indices_batch
                pair_indices_w_pref_var = zip(range(len(indices_batch)), vars)
        
        sorted_pair_indices = sorted(pair_indices_w_pref_var, key=lambda x: x[1], reverse=True)
        sorted_pairs = [(self.fragments[best_indices_batch[pair_idx[0]][0]], self.fragments[best_indices_batch[pair_idx[0]][1]]) for pair_idx in sorted_pair_indices]
        return self._down_sample_pairs(sorted_pairs)

    def preference_diff_active_learning(self, model: RewardModel, n_batches=2048):
        possible_pair_indexes = list(range(len(self.fragments)))

        fragments_list = [self.convert_traj(f) for f in self.fragments]
        fragments_tensor = th.Tensor(fragments_list)

        pred_rewards = th.sum(model.forward(fragments_tensor), dim=[1, 2]).tolist()
        calc_rewards = [sum([state['partial_rew'] for state in fragment.states]) for fragment in self.fragments]

        indices_batches = [self.shuffle_and_pair(possible_pair_indexes) for _ in range(n_batches)]
        max_diff = 0
        for indices_batch in indices_batches:
            pair_pred_probs = th.Tensor([(pred_rewards[pi[0]], pred_rewards[pi[1]]) for pi in indices_batch])
            pair_calc_probs = th.Tensor([(calc_rewards[pi[0]], calc_rewards[pi[1]]) for pi in indices_batch])

            pred_preferences = preference_prob(pair_pred_probs, 1)   
            calc_preferences = preference_prob(pair_calc_probs, 1)         
            preference_diff = abs(pred_preferences - calc_preferences)[:,0]
            
            if sum(preference_diff) > max_diff:
                max_diff = sum(preference_diff)
                best_indices_batch = indices_batch
                pair_indices_w_pref_var = zip(range(len(indices_batch)), preference_diff)
        
        sorted_pair_indices = sorted(pair_indices_w_pref_var, key=lambda x: x[1], reverse=True)
        sorted_pairs = [(self.fragments[best_indices_batch[pair_idx[0]][0]], self.fragments[best_indices_batch[pair_idx[0]][1]]) for pair_idx in sorted_pair_indices]
        return self._down_sample_pairs(sorted_pairs)

