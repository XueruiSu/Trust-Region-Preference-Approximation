# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
import time
import os
SAVE_DIR = os.environ.get("SAVE_DIR") 
EXP_NAME = os.environ.get("EXP_NAME")  

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length), Although it is a dense reward, only the last step has a reward, and the rest are 0
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # print(idx, "only has one inference", id2score[idx])
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                # print(idx, "has multi inference", id2score[idx])
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_trpa_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length), Although it is a dense reward, only the last step has a reward, and the rest are 0
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    ids = defaultdict(list)
    unique_id = 0
    bsz = scores.shape[0]
    id_tensor = torch.zeros(bsz)
    

    with torch.no_grad():
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            unique_id += 1
            ids[idx].append(unique_id)
            if len(id2score[idx]) == 1:
                raise ValueError(f"Preference Optimization must have at least 2 responses: {idx}")
        for i in range(bsz):
            id_tensor[i] = torch.tensor(ids[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores, id_tensor

def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   gamma: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for Reinforce++, operating on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # response_length = token_level_rewards.shape[-1]

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0
        
        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask
       
    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl



def compute_pairwise_policy_loss(log_prob: torch.Tensor,
                                 ref_log_prob: torch.Tensor,
                                 advantages: torch.Tensor,
                                 eos_mask: torch.Tensor,
                                 beta: float, 
                                 preference_levels: torch.Tensor,
                                 anisotropy: float = 2.5, 
                                 preference_type: str = 'top-positive', 
                                 id_tensor: torch.Tensor = None):
    agg_log_prob    = verl_F.masked_mean(log_prob, eos_mask, axis=-1)      # shape: (bs,)
    agg_ref_log_prob = verl_F.masked_mean(ref_log_prob, eos_mask, axis=-1)   # shape: (bs,)

    log_ratio = agg_log_prob - agg_ref_log_prob   # shape: (bs,)

    h_list = []
    h_high = []
    h_low = []
    
    unique_ids = torch.unique(id_tensor)
    
    for uid in unique_ids:
        group_indices = (id_tensor == uid).nonzero(as_tuple=True)[0]
        if group_indices.numel() < 2:
            continue
        
        group_pref = preference_levels[group_indices]
        print("uid, group_pref", uid, group_pref)
        
        unique_levels = torch.unique(group_pref)
        unique_levels, _ = torch.sort(unique_levels)
        
        for i in range(len(unique_levels)):
            for j in range(i+1, len(unique_levels)):
                level_i = unique_levels[i]  
                level_j = unique_levels[j]  
                
                indices_i = group_indices[(group_pref == level_i).nonzero(as_tuple=True)[0]]
                indices_j = group_indices[(group_pref == level_j).nonzero(as_tuple=True)[0]]
                
                # h = beta * [ log_ratio(y_w) - log_ratio(y_l) ]
                for idx_i in indices_i:
                    for idx_j in indices_j:
                        if level_i == 1:
                            h_val = anisotropy * beta * (log_ratio[idx_i] - log_ratio[idx_j])
                        else:
                            h_val = beta * (log_ratio[idx_i] - log_ratio[idx_j])
                        h_list.append(h_val)
                        h_high.append(log_ratio[idx_i])
                        h_low.append(log_ratio[idx_j])
    
    if len(h_list) == 0:
        print("log_ratio", log_ratio, preference_levels)
        otherStyleTime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))     
        if not os.path.exists(f"{SAVE_DIR}/{EXP_NAME}"):
            os.makedirs(f"{SAVE_DIR}/{EXP_NAME}", exist_ok=True)
            print(f"directory {SAVE_DIR}/{EXP_NAME} Create Success!")
        torch.save(log_ratio, f"{SAVE_DIR}/{EXP_NAME}/log_ratio_{otherStyleTime}_.pt")
        torch.save(preference_levels, f"{SAVE_DIR}/{EXP_NAME}/preference_levels_{otherStyleTime}_.pt")
        mask_tensor = torch.where(preference_levels == 1, torch.tensor(1.0), torch.tensor(-1.0)).to(log_ratio.device)
        loss = -torch.log(torch.sigmoid(mask_tensor * log_ratio)).mean()
        h_high_zero, h_low_zero = calc_h_high_low(log_ratio, preference_levels)
        return loss, log_ratio.mean(), h_high_zero, h_low_zero
    else:
        h_tensor = torch.stack(h_list)  # (num_pairs,)
        loss = -torch.log(torch.sigmoid(h_tensor)).mean()
        return loss, h_tensor.mean() / beta, torch.stack(h_high).mean(), torch.stack(h_low).mean()

def calc_h_high_low(log_ratio, preference_levels):
    if log_ratio[preference_levels == 1].numel() > 0:
        h_high_num = log_ratio[preference_levels == 1].numel()
        h_high_zero = (log_ratio[preference_levels == 1]).mean()
        if log_ratio.numel() == h_high_num:
            h_low_zero = (log_ratio.sum() - h_high_zero * h_high_num)
        else:
            h_low_zero = (log_ratio.sum() - h_high_zero * h_high_num) / (log_ratio.numel() - h_high_num)
    else:
        h_low_num = log_ratio[preference_levels != 1].numel()
        h_low_zero = (log_ratio[preference_levels != 1]).mean()
        if log_ratio.numel() == h_low_num:
            h_high_zero = (log_ratio.sum() - h_low_zero * h_low_num)
        else:
            h_high_zero = (log_ratio.sum() - h_low_zero * h_low_num) / (log_ratio.numel() - h_low_num)
    return h_high_zero, h_low_zero
    

def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


