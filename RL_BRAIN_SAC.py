import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import collections

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Add projection layer if dimensions differ
        self.projection = None
        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = x if self.projection is None else self.projection(x)
        out = F.relu(self.layer_norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.layer_norm2(out)
        out += residual
        out = F.relu(out)
        return out

class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(PolicyNet, self).__init__()
        # Increase network capacity
        self.fc1 = nn.Linear(state_size, hidden_size)
        
        # Use residual blocks with different dimensions
        self.res1 = ResidualBlock(hidden_size, hidden_size * 2)
        self.res2 = ResidualBlock(hidden_size * 2, hidden_size)
        
        # Add additional processing layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Use smaller initialization for output layer
        self.fc_out = nn.Linear(hidden_size, action_size)
        
        # Use orthogonal initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
            # Use smaller initialization for output layer
            if m == self.fc_out:
                m.weight.data.mul_(0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.layer_norm(self.fc2(x)))
        logits = self.fc_out(x)
        # Use temperature parameter to control the smoothness of softmax
        temperature = 1.0
        return F.softmax(logits / temperature, dim=-1)

class QValueNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        
        self.res1 = ResidualBlock(hidden_size, hidden_size * 2)
        self.res2 = ResidualBlock(hidden_size * 2, hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, action_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
            if m == self.fc_out:
                m.weight.data.mul_(0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.layer_norm(self.fc2(x)))
        return self.fc_out(x)

class RolloutBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            np.array(state),
            np.array(action, dtype=np.int64),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )

    def size(self):
        return len(self.buffer)

class SAC:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=3e-4,
            reward_decay=0.99,
            batch_size=256,
            memory_size=100000
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size
        self.memory = RolloutBuffer(memory_size)
        
        # Modify target entropy calculation
        self.target_entropy = -np.log(1.0 / n_actions) * 0.5
        
        # Add reward scaling
        self.reward_scale = 0.1
        
        # Add gradient clipping
        self.max_grad_norm = 1.0
        
        # Fix log_alpha initialization
        self.log_alpha = torch.tensor([-1.0], requires_grad=True, device=device)
        
        # Initialize networks
        self.actor = PolicyNet(n_features, n_actions).to(device)
        self.critic_1 = QValueNet(n_features, n_actions).to(device)
        self.critic_2 = QValueNet(n_features, n_actions).to(device)
        self.target_critic_1 = QValueNet(n_features, n_actions).to(device)
        self.target_critic_2 = QValueNet(n_features, n_actions).to(device)
        
        # Initialize target networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.lr * 2.0)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.lr * 2.0)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr * 0.5)
        
        # Learning rate scheduler
        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Other parameters
        self.tau = 0.005
        self.last_losses = None
        
        # Add loss tracking
        self.running_critic_loss = collections.deque(maxlen=100)
        self.running_actor_loss = collections.deque(maxlen=100)

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = self.actor(state)
            # Add temperature scaling
            temperature = max(0.1, self.log_alpha.exp().item())
            scaled_probs = F.softmax(torch.log(probs + 1e-8) / temperature, dim=-1)
            dist = torch.distributions.Categorical(scaled_probs)
            action = dist.sample()
        return action.item()

    def store_transition(self, s, a, r, s_, d):
        self.memory.add(s, a, r, s_, d)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self):
        if self.memory.size() < self.batch_size:
            return

        # Sample data and preprocess
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device) * self.reward_scale
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Calculate target Q values
        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1 = self.target_critic_1(next_states)
            next_q2 = self.target_critic_2(next_states)
            min_next_q = torch.min(next_q1, next_q2)
            # Use current alpha value to calculate target value
            alpha = self.log_alpha.exp().detach()
            v_next = (next_probs * (min_next_q - alpha * next_log_probs)).sum(dim=1, keepdim=True)
            q_target = rewards + self.gamma * (1 - dones) * v_next

        # Update Critic networks
        q1 = self.critic_1(states).gather(1, actions.unsqueeze(1))
        q2 = self.critic_2(states).gather(1, actions.unsqueeze(1))
        
        # Use Huber Loss
        critic_1_loss = F.smooth_l1_loss(q1, q_target)
        critic_2_loss = F.smooth_l1_loss(q2, q_target)

        # Update first Critic
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
        self.critic_1_optimizer.step()

        # Update second Critic
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.critic_2_optimizer.step()

        # Update Actor
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        q1_pi = self.critic_1(states).detach()
        q2_pi = self.critic_2(states).detach()
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        # Calculate actor loss
        alpha = self.log_alpha.exp().detach()
        actor_loss = (probs * (alpha * log_probs - min_q_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Update alpha
        entropy = -torch.sum(probs * log_probs, dim=1).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        # Update loss tracking
        self.running_critic_loss.append((critic_1_loss.item() + critic_2_loss.item()) / 2)
        self.running_actor_loss.append(actor_loss.item())
        
        # Update learning rate
        if len(self.running_critic_loss) >= 100:  # Wait for enough samples
            mean_critic_loss = sum(self.running_critic_loss) / len(self.running_critic_loss)
            self.actor_scheduler.step(mean_critic_loss)

        self.last_losses = {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'entropy': entropy.item()
        }

        return self.last_losses

    def get_weights(self):
        return [
            self.actor.state_dict(),
            self.critic_1.state_dict(),
            self.critic_2.state_dict()
        ]

    def set_weights(self, weights):
        self.actor.load_state_dict(weights[0])
        self.critic_1.load_state_dict(weights[1])
        self.critic_2.load_state_dict(weights[2])
        self.target_critic_1.load_state_dict(weights[1])
        self.target_critic_2.load_state_dict(weights[2])

    def save(self, checkpoint_path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.target_critic_1.load_state_dict(checkpoint['target_critic_1'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])