import time
from typing import Tuple
import os
import yaml
import torch
from torch.optim import Adam
from torch import nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

class PPO:
    def __init__(self,
                 policy_class:nn.Module,
                 obs_dim:int = 10,
                 act_dim:int = 5,
                 instance_id:str = "player",
                 hyperparameters:str = "deafult"):
        """
        create an agent for learning
        :param policy_class: insert policy class
        :param obs_dim: shape of info from env
        :param act_dim: how many possible choices for agent
        :param instance_id: id for printing and saving
        :param hyperparameters: hyperparmeters_set
        """
        self._init_hyperparameters(hyperparameters)
        self.id = instance_id

        self.PLOT_PATH = f"results/{instance_id}/ppo_learning_graph"
        self.SAVE_PATH = f"results/{instance_id}"

        if not os.path.exists(f'results/{self.id}'):
            os.makedirs(f'results/{self.id}')

        self.best_mean_rew: float =        -999999.0
        self.actor_state_dict_from_best =  None
        self.critic_state_dict_from_best = None

        self.avg_rew_hist:np.ndarray =          np.array([])
        self.avg_actor_losses_hist:np.ndarray = np.array([])

        self.batch_obs =       []
        self.batch_acts =      []
        self.batch_log_probs = []
        self.batch_rews =      []
        self.ep_rews =         []
        self.batch_masks =     []

        self.continues_env:bool = False
        self.obs_dim:int =        obs_dim
        self.act_dim:int =        act_dim

        self.i_so_far = 0

        self.actor =  policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1)
        self.actor_optim:torch.optim.optimizer.Optimizer =  Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim:torch.optim.optimizer.Optimizer = Adam(self.critic.parameters(), lr=self.lr)

        if self.continues_env:
            self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
            self.cov_mat = torch.diag(self.cov_var)

        self.logger:dict = {
            'delta_t':      time.time_ns(),
            'batch_rews':   [],
            'actor_losses': []
        }

    def learn_step(self):
        """
        actualise policy
        """
        self.to_tensors()

        v, _ ,_ = self.evaluate(self.batch_obs, self.batch_acts.flatten())

        a_k = self.batch_rtgs - v.detach()

        a_k = (a_k - a_k.mean()) / (a_k.std() + 1e-10) if a_k.std()> 1e-6 else a_k - a_k.mean()

        for _ in range(self.n_updates_per_iteration):
            if len(self.batch_obs) ==0:
                return
            v, curr_log_prob, ent = self.evaluate(self.batch_obs, self.batch_acts.flatten())
            self.fit(ent,v,curr_log_prob,a_k)


        self.save()
        self.clear()

    def learn_step_masked(self):
        """
        actualize policy in masked env
        """
        self.to_tensors()
        batch_masks = torch.tensor(data=np.array(self.batch_masks), dtype=torch.bool)

        v, _, _ = self.evaluate_masked(self.batch_obs, self.batch_acts, batch_masks)

        a_k = self.batch_rtgs - v.detach()

        a_k = (a_k - a_k.mean()) / (a_k.std() + 1e-10) if a_k.std() > 1e-6 else a_k - a_k.mean()
        for _ in range(self.n_updates_per_iteration):
            if len(self.batch_obs) == 0:
                return
            v, curr_log_prob, ent = self.evaluate_masked(self.batch_obs, self.batch_acts, batch_masks)

            self.fit(ent, v, curr_log_prob, a_k)

        self.save()
        self.clear()

    def to_tensors(self):
        """
        make every data for learning tensor
        """
        self.batch_obs = torch.tensor(np.stack(self.batch_obs), dtype=torch.float32)
        self.batch_log_probs = torch.tensor(np.stack(self.batch_log_probs), dtype=torch.int32)
        self.batch_acts = torch.tensor(np.array(self.batch_acts), dtype=torch.long)
        self.batch_rtgs = self.compute_rtgs(self.batch_rews)

    def fit(self,ent,v,curr_log_prob,a_k):
        """
        calculate loss and make the model right

        :param ent: the entropy from model
        :param v: value
        :param curr_log_prob:curr log prob calculate ratio
        :param a_k: advantage batch_rtgs - v
        """
        ratio = torch.exp(curr_log_prob - self.batch_log_probs)

        surr1 = a_k * ratio
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a_k
        ent =   -ent.mean()

        actor_loss = (-torch.min(surr1, surr2)).mean() + self.exploration * ent
        critic_loss = nn.MSELoss()(v.squeeze(), self.batch_rtgs.squeeze())

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.parameters_max_change)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.parameters_max_change)
        self.critic_optim.step()

        self.logger['actor_losses'].append(actor_loss.detach())

    def save(self):
        """
        save model
        """
        self._log_summary()

        self.i_so_far += 1

        if self.i_so_far % self.save_freq == 0:
            torch.save(self.actor.state_dict(), self.SAVE_PATH + f'ppo_actor.pth')
            torch.save(self.critic.state_dict(), self.SAVE_PATH + f'ppo_critic.pth')

    def clear(self):
        """
        set every thing need to learning to empty
        :return:
        """
        self.batch_rews = []
        self.batch_obs = []
        self.batch_log_probs = []
        self.batch_acts = []
        self.batch_masks = []



    def act(self, env):
        """
        make action in env and colect the data
        :param env: env for doing step
        """
        obs, rew, terminated, truncated, info = env.last(observe=True)
        done = terminated or truncated

        if done and len(self.ep_rews) > 0:
            self.ep_rews[-1] += rew

        if not done:
            self.batch_obs.append(obs.flatten())

            action, log_prob = self.get_action(obs.flatten())

            self.ep_rews.append(rew)
            self.batch_acts.append(action)
            self.batch_log_probs.append(log_prob)
            env.step(action)
        else:
            env.step(None)

        if self.render:
            env.render()

    def act_masked(self, env):
        """
        make action in masked env and colect the data
        :param env: env for doing step
        """
        obs, rew, terminated, truncated, info = env.last(observe=True)
        done = terminated or truncated

        if done and len(self.ep_rews) > 0:
            self.ep_rews[-1] += rew

        if not done:
            self.batch_obs.append(obs['observation'])

            action, log_prob = self.get_action_masked(obs['observation'],obs['action_mask'])

            self.ep_rews.append(rew)
            self.batch_acts.append(action)
            self.batch_log_probs.append(log_prob)
            env.step(action)
        else:
            env.step(None)

        if self.render:
            env.render()

    def chose_action(self,env):
        """
        pick action
        :param env: env for doing step
        """
        obs, rew, terminated, truncated, info = env.last(observe=True)
        done = terminated or truncated

        if not done:
            action, log_prob = self.get_action(obs)
            env.step(action)
        else:
            env.step(None)
        env.render()

    def chose_action_masked(self,env):
        """
        pick action from maksed env
        :param env: env for doing step
        """
        obs, rew, terminated, truncated, info = env.last(observe=True)
        done = terminated or truncated

        if not done:
            action, log_prob = self.get_action_masked(obs['observation'], obs['action_mask'])
            env.step(action)
        else:
            env.step(None)
        env.render()

    def end_of_game(self):
        """
        add ep_rew to batch_rew
        """
        self.batch_rews.append(self.ep_rews)
        self.ep_rews = []

    def compute_rtgs(self, batch_rews) -> torch.Tensor:
        """
        compute rtgs with formula x[n] = rews[n] + gamma * x[n-1]
        :param batch_rews: batch of rews from games
        :return: batch_rtgs
        """
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discount_factor = 0
            for rew in reversed(ep_rews):
                discount_factor = rew + self.gamma * discount_factor
                batch_rtgs.insert(0, discount_factor)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def get_action(self, obs) -> Tuple[np.ndarray, torch.Tensor]:
        """
        fet action from policy
        :param obs: observation of shape 1 obs_dm
        :return: action and log prob for action
        """
        mean = self.actor(obs)
        if not self.continues_env:
            dist = Categorical(logits=mean)
        else:
            dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def get_action_masked(self, obs, mask) -> Tuple[np.ndarray, torch.Tensor]:
        """
        fet action from policy in masekd env
        :param obs: observation of shape 1 obs_dm
        :param mask: mask for action
        :return: action and log prob for action
        """
        mean = self.actor(obs)
        if not self.continues_env:
            mask = torch.tensor(mask,dtype=torch.bool)
            mask_mean = mean.masked_fill(~mask,-1e8)
            self.batch_masks.append(mask.numpy())
            dist = Categorical(logits= mask_mean)
        else:
            dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        evalute policy and critic
        :param batch_obs: batch with obs from games
        :param batch_acts: batch with acts from games
        :return: value , curr_log_prob, entropy of policy
        """
        v = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        if not self.continues_env:
            dist = Categorical(logits=mean)
        else:
            dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(batch_acts)
        ent = dist.entropy()
        return v, log_prob,ent

    def evaluate_masked(self, batch_obs, batch_acts, batch_masks) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        evalute policy and critic  in env with masks
        :param batch_obs: batch with obs from games
        :param batch_acts: batch with acts from games
        :return: value , curr_log_prob, entropy of policy
        """
        v = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        if not self.continues_env:
            masked_mean = mean.masked_fill(~batch_masks,-1e8)
            dist = Categorical(logits=masked_mean)
        else:
            dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(batch_acts)
        ent = dist.entropy()
        return v, log_prob,ent

    def _init_hyperparameters(self, hyperparameters:str) -> None:
        with open("hyperparmeters.yml","r") as f:
            all_hyperparameters_sets = yaml.safe_load(f)
            hyper = all_hyperparameters_sets[hyperparameters]
        self.lr:float = hyper['lr']
        self.save_freq:int = hyper['save_freq']
        self.n_updates_per_iteration:int = hyper['n_updates_per_iteration']
        self.clip:float = hyper['clip']
        self.render:bool = hyper['render']
        self.gamma:float = hyper['gamma']
        self.seed:int|None = hyper['seed']
        self.exploration:float = hyper['exploration']
        self.parameters_max_change:float = hyper['parameters_max_change']

        if type(self.seed) == int:
            torch.manual_seed(self.seed)
            print("seed set")

    def _log_summary(self) -> None:
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))
        avg_ep_rews = np.mean([np.sum(ep_reward) for ep_reward in self.batch_rews])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        self.avg_rew_hist = np.append(self.avg_rew_hist, avg_ep_rews)
        self.avg_actor_losses_hist = np.append(self.avg_actor_losses_hist, -avg_actor_loss)

        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print(flush=True)
        print(f"my id is {self.id}")
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print("Model State Dict save")
        print(flush=True)

        self.plot_graph(self.avg_rew_hist, self.avg_actor_losses_hist)

        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    def plot_graph(self, hist_avg_rew, hist_avg_act_loss) -> None:
        fig = plt.figure(figsize=(8,5))
        plt.subplot(121)
        plt.ylabel("avg_actor_loss")
        plt.xlabel("epoch")
        plt.plot(hist_avg_act_loss)

        plt.subplot(122)
        plt.ylabel("avg_rew")
        plt.xlabel("epoch")
        plt.grid(axis="y")
        plt.plot(hist_avg_rew, color='orange')

        plt.subplots_adjust(wspace=1.0)
        fig.savefig(self.PLOT_PATH)
        plt.close(fig)

    def load(self) -> None:
        try:
            self.actor.load_state_dict(torch.load(self.SAVE_PATH+"ppo_actor.pth"))
            self.critic.load_state_dict(torch.load(self.SAVE_PATH+"ppo_critic.pth"))
        except FileNotFoundError:
            raise "that dont exist"
    def __getattr__(self, item:str):
        raise AttributeError(
            f"Item '{item}' does not exist. If you want help, call the help method."
        )
#podtlenem