from torch import nn
from torch import Tensor


class Model:
    def __init__(self, policy_logits=780, hidden_state_size=32):
        self.policy_logits = policy_logits
        self.hidden_state_size = hidden_state_size

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def init_infer(self, obs):
        value = np.random.rand(1)
        reward = np.random.rand(1)
        policy_logits = np.random.rand(self.policy_logits)
        hidden_state = np.random.rand(self.hidden_state_size)
        return value, reward, self.softmax(policy_logits), hidden_state

    def infer(self, obs, o=None):
        if obs is not None:
            if len(obs.shape) > 1:
                batch = obs.shape[0]
                value = np.random.rand(batch)
                reward = np.random.rand(batch)
                policy_logits = np.random.rand(batch, self.policy_logits)
                hidden_state = np.random.rand(batch, self.hidden_state_size)
                return value, reward, self.softmax(policy_logits), hidden_state
        value = np.random.rand(1)
        reward = np.random.rand(1)
        policy_logits = np.random.rand(self.policy_logits)
        hidden_state = np.random.rand(self.hidden_state_size)
        return value, reward, self.softmax(policy_logits), hidden_state
