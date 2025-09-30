import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

π = np.pi

class SpriteQLearning:
    """
    Sprite Q-Learning: Creative exploration through counterfactual rotation
    
    Key innovations:
    1. Complex-valued Q-functions with amplitude and phase
    2. Kramers escape dynamics for switching policies
    3. Phase accumulation in counterfactual dimensions
    4. Surprise-gated rotation preventing premature convergence
    """
    
    def __init__(self, n_states, n_actions, 
                 α=0.1, γ=0.99, β=0.1, κ=0.2,
                 phase_threshold=π/4, kramers_ω=1.0):
        """
        Parameters:
        -----------
        α : learning rate for amplitude (standard Q-learning)
        γ : discount factor
        β : rotation rate for phase (counterfactual strength)
        κ : crystallization factor (phase → amplitude coupling)
        phase_threshold : trigger for crystallization
        kramers_ω : attempt frequency for escape
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.α = α
        self.γ = γ
        self.β = β
        self.κ = κ
        self.phase_threshold = phase_threshold
        self.kramers_ω = kramers_ω
        
        # Complex Q-values: Ψ(s,a) = |Ψ| * e^{iφ}
        self.amplitude = np.ones((n_states, n_actions)) / np.sqrt(n_actions)
        self.phase = np.random.uniform(-np.pi, np.pi, (n_states, n_actions))
        
        # Model for surprise computation
        self.reward_model = defaultdict(lambda: defaultdict(float))
        self.transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        
        # Tracking
        self.surprise_history = []
        self.escape_history = []
        
    def get_policy(self, state):
        """Extract policy from amplitude: p(a|s) = |Ψ(s,a)|²"""
        probs = self.amplitude[state]**2
        return probs / probs.sum()
    
    def select_action(self, state, mode='sample'):
        """
        Action selection with two modes:
        - 'sample': Sample from current policy (for acting)
        - 'greedy': Take argmax (for evaluation)
        """
        policy = self.get_policy(state)
        
        if mode == 'sample':
            return np.random.choice(self.n_actions, p=policy)
        else:  # greedy
            return np.argmax(policy)
    
    def compute_surprise(self, state, action, reward, next_state):
        """
        Measure surprise as prediction error in both reward and transition
        Scaled by uncertainty √(p(1-p))
        """
        # Reward surprise
        expected_reward = self.reward_model[state][action]
        reward_surprise = abs(reward - expected_reward)
        
        # Transition surprise (negative log likelihood)
        total_visits = sum(self.transition_counts[state][action].values())
        if total_visits > 0:
            p_transition = self.transition_counts[state][action][next_state] / total_visits
            transition_surprise = -np.log(p_transition + 1e-10)
        else:
            transition_surprise = np.log(self.n_states)  # maximum surprise
        
        # Weight by uncertainty at this action
        p_a = self.get_policy(state)[action]
        uncertainty = np.sqrt(p_a * (1 - p_a))
        
        # Combined surprise (weighted)
        surprise = (reward_surprise + 0.5 * transition_surprise) * uncertainty
        
        return surprise, uncertainty
    
    def kramers_escape_check(self, surprise, uncertainty):
        """
        Kramers escape: should we rotate to counterfactuals?
        
        Escape rate: k = ω * exp(-ΔU/kT)
        where ΔU = surprise (barrier height)
              kT = uncertainty (thermal energy)
        """
        barrier = surprise
        temperature = uncertainty + 1e-3
        
        escape_rate = self.kramers_ω * np.exp(-barrier / temperature)
        escape_prob = 1 - np.exp(-escape_rate * 0.1)  # dt = 0.1
        
        should_escape = np.random.random() < escape_prob
        
        return should_escape, escape_prob
    
    def update_phase(self, state, action, surprise, should_rotate):
        """
        Phase rotation: accumulate counterfactual potential
        
        When surprise is high, rotate phase toward alternative actions.
        Phase doesn't immediately affect policy—it's "latent exploration."
        """
        if not should_rotate:
            # Just random diffusion
            self.phase[state, action] += 0.01 * np.random.randn()
            return
        
        # Current phase
        φ_current = self.phase[state, action]
        
        # Target: mean phase of other actions (counterfactuals)
        other_actions = [a for a in range(self.n_actions) if a != action]
        φ_others = np.array([self.phase[state, a] for a in other_actions])
        
        # Complex mean (geometric center on unit circle)
        complex_mean = np.mean(np.exp(1j * φ_others))
        φ_target = np.angle(complex_mean)
        
        # Rotate toward target
        Δφ = self.β * np.sin(φ_target - φ_current)
        self.phase[state, action] += Δφ
        
        # Also perturb other actions (create diversity)
        for a in other_actions:
            perturbation = self.β * 0.5 * np.cos(self.phase[state, a] - φ_current)
            self.phase[state, a] += perturbation
        
        # Wrap to [-π, π]
        self.phase[state] = np.angle(np.exp(1j * self.phase[state]))
    
    def crystallize_phase(self, state):
        """
        Crystallization: convert accumulated phase into amplitude changes
        
        When phase diverges significantly from mean, it "crystallizes" into
        actual exploration by temporarily boosting that action's probability.
        
        This is the Kramers escape moment—jumping to a counterfactual.
        """
        φ_mean = np.mean(self.phase[state])
        phase_deviations = np.abs(self.phase[state] - φ_mean)
        
        crystallized = False
        
        for action in range(self.n_actions):
            if phase_deviations[action] > self.phase_threshold:
                # Crystallize this action
                boost = 1 + self.κ * phase_deviations[action]
                self.amplitude[state, action] *= boost
                
                # Reset phase after crystallization
                self.phase[state, action] = φ_mean
                
                crystallized = True
        
        if crystallized:
            # Renormalize amplitudes to maintain probability conservation
            amp_squared = self.amplitude[state]**2
            self.amplitude[state] = np.sqrt(amp_squared / amp_squared.sum())
    
    def update(self, state, action, reward, next_state, done):
        """
        Full Sprite RL update: L-E-R cycle
        
        L (Log): Measure surprise
        E (Exp): Standard TD update on amplitude
        R (Rotate): Phase rotation in counterfactual space
        """
        # Update models for surprise computation
        self.reward_model[state][action] += 0.1 * (reward - self.reward_model[state][action])
        self.transition_counts[state][action][next_state] += 1
        self.visit_counts[state][action] += 1
        
        # L: Log (measure surprise)
        surprise, uncertainty = self.compute_surprise(state, action, reward, next_state)
        self.surprise_history.append(surprise)
        
        # E: Exp (standard TD update on amplitude)
        if not done:
            target_value = (self.amplitude[next_state]**2).max()
        else:
            target_value = 0
        
        current_value = self.amplitude[state, action]**2
        td_error = reward + self.γ * target_value - current_value
        
        # Update amplitude (standard Q-learning)
        amplitude_update = self.α * td_error / (2 * self.amplitude[state, action] + 1e-8)
        self.amplitude[state, action] += amplitude_update
        
        # Ensure non-negative
        self.amplitude[state, action] = max(self.amplitude[state, action], 0.01)
        
        # Renormalize
        amp_squared = self.amplitude[state]**2
        self.amplitude[state] = np.sqrt(amp_squared / amp_squared.sum())
        
        # R: Rotate (phase dynamics)
        should_rotate, escape_prob = self.kramers_escape_check(surprise, uncertainty)
        self.escape_history.append(escape_prob)
        
        self.update_phase(state, action, surprise, should_rotate)
        
        # Crystallization check
        self.crystallize_phase(state)
    
    def get_diagnostics(self):
        """Return diagnostic information about learning dynamics"""
        return {
            'mean_surprise': np.mean(self.surprise_history[-100:]) if self.surprise_history else 0,
            'mean_escape_prob': np.mean(self.escape_history[-100:]) if self.escape_history else 0,
            'policy_entropy': self._compute_policy_entropy(),
            'phase_variance': np.var(self.phase)
        }
    
    def _compute_policy_entropy(self):
        """Average entropy across all states (measures exploration)"""
        entropies = []
        for s in range(self.n_states):
            p = self.get_policy(s)
            entropy = -np.sum(p * np.log(p + 1e-10))
            entropies.append(entropy)
        return np.mean(entropies)


# Example: Creative Maze Navigation
class CreativeMaze:
    """
    A maze where creative exploration is rewarded.
    
    Standard path: Easy but low reward
    Creative path: Requires exploring unlikely states but high reward
    """
    def __init__(self):
        self.n_states = 25  # 5x5 grid
        self.n_actions = 4  # up, down, left, right
        self.goal_standard = 24  # bottom-right
        self.goal_creative = 4   # top-right (requires backtracking)
        self.reset()
    
    def reset(self):
        self.state = 0  # top-left
        return self.state
    
    def step(self, action):
        # Grid navigation
        row, col = self.state // 5, self.state % 5
        
        if action == 0 and row > 0:  # up
            row -= 1
        elif action == 1 and row < 4:  # down
            row += 1
        elif action == 2 and col > 0:  # left
            col -= 1
        elif action == 3 and col < 4:  # right
            col += 1
        
        self.state = row * 5 + col
        
        # Rewards
        if self.state == self.goal_standard:
            return self.state, 1.0, True
        elif self.state == self.goal_creative:
            return self.state, 10.0, True  # Much better!
        else:
            return self.state, -0.01, False  # Small step cost


# Training comparison
def train_and_compare(n_episodes=1000):
    """Compare Sprite RL vs standard Q-learning"""
    
    env = CreativeMaze()
    
    # Sprite RL agent
    sprite_agent = SpriteQLearning(
        n_states=env.n_states,
        n_actions=env.n_actions,
        β=0.15,  # Moderate rotation
        κ=0.3    # Strong crystallization
    )
    
    sprite_returns = []
    sprite_creative_discoveries = 0
    
    print("Training Sprite RL Agent...")
    for ep in range(n_episodes):
        state = env.reset()
        episode_return = 0
        
        for step in range(100):
            action = sprite_agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            sprite_agent.update(state, action, reward, next_state, done)
            
            episode_return += reward
            state = next_state
            
            if done:
                if next_state == env.goal_creative:
                    sprite_creative_discoveries += 1
                break
        
        sprite_returns.append(episode_return)
        
        if ep % 100 == 0:
            diag = sprite_agent.get_diagnostics()
            print(f"Episode {ep}: Return={episode_return:.2f}, "
                  f"Entropy={diag['policy_entropy']:.3f}, "
                  f"Creative discoveries={sprite_creative_discoveries}")
    
    return sprite_returns, sprite_creative_discoveries


if __name__ == "__main__":
    returns, discoveries = train_and_compare()
    print(f"\nFinal: {discoveries} creative path discoveries")
    print(f"Mean return (last 100 episodes): {np.mean(returns[-100:]):.2f}")
