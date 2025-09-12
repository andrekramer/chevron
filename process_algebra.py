import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns

class ProcessAlgebra:
    """
    Mathematical framework for composing different types of processes
    using log/exp transformations to make multiplication tractable
    """
    
    def __init__(self, dt=0.01, T=10.0):
        self.dt = dt
        self.T = T
        self.t = np.arange(0, T, dt)
        self.n_steps = len(self.t)
    
    def simple_random(self, scale=1.0):
        """
        Simple Random (Q3): Markovian + Undirected
        Standard Brownian motion - memoryless random walk
        """
        dW = np.random.randn(self.n_steps) * np.sqrt(self.dt) * scale
        W = np.cumsum(dW)
        return W
    
    def simple_adaptive(self, target=0.0, strength=1.0, x0=5.0):
        """
        Simple Adaptive (Q2): Markovian + Directed  
        Ornstein-Uhlenbeck process - mean reversion toward target
        dx = -strength * (x - target) * dt + noise
        """
        x = np.zeros(self.n_steps)
        x[0] = x0
        
        for i in range(1, self.n_steps):
            drift = -strength * (x[i-1] - target)
            noise = np.random.randn() * np.sqrt(self.dt) * 0.5
            x[i] = x[i-1] + drift * self.dt + noise
            
        return x
    
    def complex_random(self, alpha=0.8, scale=1.0):
        """
        Complex Random (Q4): Non-Markovian + Undirected
        Fractional Brownian motion - long-range correlations but no direction
        """
        # Generate fractional Brownian motion using FFT method
        n = self.n_steps
        r = np.zeros(2 * n)
        
        # Covariance function for fBm
        for k in range(2 * n):
            if k == 0:
                r[k] = 1
            else:
                r[k] = 0.5 * (abs(k-1)**(2*alpha) - 2*abs(k)**(2*alpha) + abs(k+1)**(2*alpha))
        
        # Generate using circulant embedding
        r_fft = np.fft.fft(r)
        z = np.random.randn(2 * n) + 1j * np.random.randn(2 * n)
        fBm = np.fft.ifft(np.sqrt(r_fft) * z)[:n]
        
        return np.real(fBm) * scale
    
    def complex_adaptive_direct(self, memory_weight=0.7, target_strength=1.0, target=0.0):
        """
        Complex Adaptive (Q1): Non-Markovian + Directed
        Direct implementation - uses weighted history to drive toward target
        """
        x = np.zeros(self.n_steps)
        x[0] = np.random.randn()
        
        for i in range(1, self.n_steps):
            # Non-Markovian: weighted sum of history
            if i > 10:
                weights = np.exp(-np.arange(i) * 0.1)  # exponential decay
                weights /= weights.sum()
                memory_term = np.dot(weights, x[:i])
            else:
                memory_term = np.mean(x[:i])
            
            # Directed: drive toward target based on memory
            drift = -target_strength * (memory_term - target)
            noise = np.random.randn() * 0.3
            
            x[i] = x[i-1] + drift * self.dt + noise * np.sqrt(self.dt)
        
        return x
    
    def compose_via_log_exp(self, simple_adaptive_process, complex_random_process, 
                           mixing_param=0.5, base_offset=10.0):
        """
        Compose Simple Adaptive + Complex Random → Complex Adaptive
        Using log/exp trick to make multiplication tractable
        
        Key insight: 
        log(Complex_Adaptive) = α * log(Simple_Adaptive) + β * log(Complex_Random)
        where processes are shifted to be positive
        """
        # Shift processes to be positive (needed for log)
        sa_positive = simple_adaptive_process - simple_adaptive_process.min() + base_offset
        cr_positive = complex_random_process - complex_random_process.min() + base_offset
        
        # Take logs
        log_sa = np.log(sa_positive)
        log_cr = np.log(cr_positive)
        
        # Linear combination in log space
        log_composed = mixing_param * log_sa + (1 - mixing_param) * log_cr
        
        # Exponiate back
        composed = np.exp(log_composed)
        
        # Re-center
        composed = composed - np.mean(composed)
        
        return composed
    
    def analyze_properties(self, process, name="Process"):
        """Analyze temporal properties of a process"""
        # Autocorrelation
        autocorr = np.correlate(process - np.mean(process), 
                               process - np.mean(process), 'full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Memory decay (improved method)
        lags = np.arange(len(autocorr)) * self.dt
        # Find where autocorrelation drops to 1/e ≈ 0.37 instead of 0.1
        memory_decay = np.where(autocorr < np.exp(-1))[0]
        memory_time = lags[memory_decay[0]] if len(memory_decay) > 0 else self.T
        
        # Directedness (trend strength)
        trend = np.polyfit(self.t, process, 1)[0]
        
        # Complexity (approximate entropy)
        def approx_entropy(data, m=2, r=0.2):
            N = len(data)
            patterns = {}
            
            for i in range(N - m + 1):
                pattern = tuple(data[i:i+m])
                if pattern in patterns:
                    patterns[pattern] += 1
                else:
                    patterns[pattern] = 1
            
            entropy = 0
            for count in patterns.values():
                prob = count / (N - m + 1)
                if prob > 0:
                    entropy -= prob * np.log(prob)
            
            return entropy
        
        complexity = approx_entropy(process)
        
        return {
            'memory_time': memory_time,
            'trend_strength': abs(trend),
            'complexity': complexity,
            'autocorr': autocorr[:100]  # First 100 lags
        }

def demonstrate_quadrant_algebra():
    """Demonstrate the composition of processes using log/exp algebra"""
    
    pa = ProcessAlgebra(dt=0.01, T=50.0)  # Longer time series for better statistics
    
    # Generate base processes
    simple_rand = pa.simple_random(scale=1.0)
    simple_adapt = pa.simple_adaptive(target=0.0, strength=2.0, x0=3.0)
    complex_rand = pa.complex_random(alpha=0.8, scale=2.0)
    complex_adapt_direct = pa.complex_adaptive_direct()
    
    # Compose using log/exp algebra
    complex_adapt_composed = pa.compose_via_log_exp(simple_adapt, complex_rand, mixing_param=0.6)
    
    # Analyze all processes
    processes = {
        'Simple Random (Q3)': simple_rand,
        'Simple Adaptive (Q2)': simple_adapt, 
        'Complex Random (Q4)': complex_rand,
        'Complex Adaptive (Direct)': complex_adapt_direct,
        'Complex Adaptive (Composed)': complex_adapt_composed
    }
    
    analysis = {}
    for name, proc in processes.items():
        analysis[name] = pa.analyze_properties(proc, name)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quadrant Algebra: Process Composition via Log/Exp', fontsize=16, fontweight='bold')
    
    # Plot time series
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, (name, proc) in enumerate(processes.items()):
        ax = axes[0, i % 3] if i < 3 else axes[1, (i-3) % 2]
        ax.plot(pa.t, proc, color=colors[i], linewidth=1.5)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Plot autocorrelation comparison
    axes[1, 2].set_title('Autocorrelation Comparison', fontweight='bold')
    for i, (name, proc) in enumerate(processes.items()):
        lags = np.arange(100) * pa.dt
        axes[1, 2].plot(lags, analysis[name]['autocorr'], 
                       color=colors[i], label=name, linewidth=2)
    axes[1, 2].set_xlabel('Lag (time)')
    axes[1, 2].set_ylabel('Autocorrelation')
    axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("QUADRANT ALGEBRA ANALYSIS")
    print("=" * 50)
    print(f"{'Process':<25} {'Memory Time':<12} {'Trend':<10} {'Complexity':<10}")
    print("-" * 50)
    
    for name, props in analysis.items():
        print(f"{name:<25} {props['memory_time']:<12.2f} {props['trend_strength']:<10.3f} {props['complexity']:<10.2f}")
    
    print("\nCOMPOSITION INSIGHT:")
    print("Complex Adaptive (Composed) should exhibit:")
    print("- Memory time between Simple Adaptive and Complex Random")
    print("- Trend strength influenced by Simple Adaptive component") 
    print("- Complexity influenced by Complex Random component")
    
    # Verify the composition hypothesis
    sa_memory = analysis['Simple Adaptive (Q2)']['memory_time']
    cr_memory = analysis['Complex Random (Q4)']['memory_time'] 
    composed_memory = analysis['Complex Adaptive (Composed)']['memory_time']
    
    print(f"\nVERIFICATION:")
    print(f"Simple Adaptive memory time: {sa_memory:.2f}")
    print(f"Complex Random memory time: {cr_memory:.2f}")
    print(f"Composed memory time: {composed_memory:.2f}")
    print(f"Prediction check: {sa_memory:.2f} < {composed_memory:.2f} < {cr_memory:.2f}? {sa_memory < composed_memory < cr_memory}")
    
    return processes, analysis

def mathematical_framework():
    """
    Theoretical foundation for quadrant algebra
    """
    print("MATHEMATICAL FRAMEWORK FOR QUADRANT ALGEBRA")
    print("=" * 50)
    
    print("\n1. PROCESS REPRESENTATION:")
    print("   Simple Random (Q3):    X(t) ~ Wiener process")
    print("   Simple Adaptive (Q2):  X(t) ~ Ornstein-Uhlenbeck")  
    print("   Complex Random (Q4):   X(t) ~ Fractional Brownian Motion")
    print("   Complex Adaptive (Q1): X(t) ~ Memory-driven target pursuit")
    
    print("\n2. COMPOSITION RULE:")
    print("   log(Q1) = α·log(Q2) + β·log(Q4) + noise")
    print("   where α + β ≈ 1, and processes are shifted positive")
    
    print("\n3. INTERPRETATION:")
    print("   - α controls how much 'directedness' from Q2")
    print("   - β controls how much 'memory complexity' from Q4") 
    print("   - Log space makes multiplicative effects additive")
    print("   - Exp back to get geometric mean-like combination")
    
    print("\n4. BIOLOGICAL ANALOGY:")
    print("   Q2 component: Homeostatic regulation (feedback control)")
    print("   Q4 component: Historical context & environmental memory")
    print("   Q1 result: Adaptive learning with rich temporal structure")
    
    print("\n5. MACHINE LEARNING PARALLEL:")
    print("   Q2: Gradient descent (optimization pressure)")
    print("   Q4: Exploration noise with temporal correlations")  
    print("   Q1: Learning dynamics (directed exploration)")

def validate_composition_directly():
    """
    Direct validation: Does log(Q1) ≈ α·log(Q2) + β·log(Q4)?
    This tests the composition hypothesis directly rather than through emergent properties
    """
    
    pa = ProcessAlgebra(dt=0.01, T=50.0)
    
    # Generate base processes
    simple_adapt = pa.simple_adaptive(target=0.0, strength=1.5, x0=2.0)
    complex_rand = pa.complex_random(alpha=0.3, scale=1.5)
    
    # Compose via our algebra
    α, β = 0.6, 0.4
    offset = 5.0  # Ensure positivity
    
    sa_pos = simple_adapt - simple_adapt.min() + offset
    cr_pos = complex_rand - complex_rand.min() + offset
    
    # The composition rule
    log_sa = np.log(sa_pos)
    log_cr = np.log(cr_pos)
    log_composed_predicted = α * log_sa + β * log_cr
    composed_predicted = np.exp(log_composed_predicted)
    
    # Also generate a "true" complex adaptive process for comparison
    complex_adapt_direct = pa.complex_adaptive_direct()
    ca_pos = complex_adapt_direct - complex_adapt_direct.min() + offset
    log_ca_actual = np.log(ca_pos)
    
    # Test the composition hypothesis
    correlation_predicted_actual = np.corrcoef(log_composed_predicted, log_ca_actual)[0,1]
    
    # Test linear relationship in log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        α * log_sa + β * log_cr, log_ca_actual
    )
    
    print("DIRECT COMPOSITION VALIDATION")
    print("=" * 50)
    print(f"Composition rule: log(Q1) = {α}·log(Q2) + {β}·log(Q4)")
    print(f"Correlation between predicted and actual log processes: {correlation_predicted_actual:.3f}")
    print(f"Linear regression: slope={slope:.3f}, R²={r_value**2:.3f}, p={p_value:.3e}")
    
    # Test decomposition: can we recover α, β from a known composition?
    try:
        # Stack the log processes as predictors
        X = np.column_stack([log_sa, log_cr])
        y = log_composed_predicted
        
        # Solve for coefficients
        coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        recovered_α, recovered_β = coefficients
        
        print(f"\nDECOMPOSITION TEST:")
        print(f"Original α={α:.3f}, β={β:.3f}")
        print(f"Recovered α={recovered_α:.3f}, β={recovered_β:.3f}")
        print(f"Recovery error: α={abs(α-recovered_α):.4f}, β={abs(β-recovered_β):.4f}")
        
    except Exception as e:
        print(f"Decomposition failed: {e}")
    
    # Visualize the relationship
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(pa.t, log_sa, 'g-', alpha=0.7, label='log(Q2)')
    plt.plot(pa.t, log_cr, 'r-', alpha=0.7, label='log(Q4)')
    plt.plot(pa.t, log_composed_predicted, 'b-', linewidth=2, label=f'{α}·log(Q2) + {β}·log(Q4)')
    plt.title('Log-Space Composition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(log_composed_predicted, log_ca_actual, alpha=0.6, s=1)
    plt.plot([log_composed_predicted.min(), log_composed_predicted.max()], 
             [log_ca_actual.min(), log_ca_actual.max()], 'r--', alpha=0.8)
    plt.xlabel('Predicted log(Q1)')
    plt.ylabel('Actual log(Q1)')
    plt.title(f'Correlation: {correlation_predicted_actual:.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(pa.t, composed_predicted, 'b-', linewidth=2, label='Composed Q1')
    plt.plot(pa.t, complex_adapt_direct, 'purple', alpha=0.7, label='Direct Q1')
    plt.title('Exponiated Processes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'correlation': correlation_predicted_actual,
        'regression_r2': r_value**2,
        'recovered_alpha': recovered_α if 'recovered_α' in locals() else None,
        'recovered_beta': recovered_β if 'recovered_β' in locals() else None
    }

# Add this to the main execution
if __name__ == "__main__":
    from scipy import stats
    
    # Run original analysis
    mathematical_framework()
    print("\n" + "="*60 + "\n")
    processes, analysis = demonstrate_quadrant_algebra()
    
    print("\n" + "="*60 + "\n")
    # Run direct validation
    validation_results = validate_composition_directly()
