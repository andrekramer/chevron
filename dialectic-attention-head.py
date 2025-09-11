import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

class DialecticalAttentionHead(nn.Module):
    """
    Dialectical attention head that uses opposing views to iteratively refine token representations.
    
    Key features:
    - Two opposing lenses (thesis/antithesis) for context summarization
    - Iterative refinement with early stopping based on stability
    - Learned opposition axes rather than hardcoded splits
    - Efficient computation by reusing attention weights
    """
    
    def __init__(
        self,
        d_model: int,
        d_head: int = None,
        max_rounds: int = 3,
        stability_threshold: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_head or d_model // 8
        self.max_rounds = max_rounds
        self.stability_threshold = stability_threshold
        
        # Standard attention components
        self.q_proj = nn.Linear(d_model, self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_head, bias=False)
        
        # Dialectical components
        # Learn two opposing transformation matrices for thesis/antithesis
        self.thesis_proj = nn.Linear(self.d_head, self.d_head)
        self.antithesis_proj = nn.Linear(self.d_head, self.d_head)
        
        # Tension measurement network
        self.tension_net = nn.Sequential(
            nn.Linear(self.d_head * 2, self.d_head),
            nn.ReLU(),
            nn.Linear(self.d_head, 1),
            nn.Sigmoid()
        )
        
        # Synthesis network for combining opposing views
        self.synthesis_net = nn.Sequential(
            nn.Linear(self.d_head * 3, self.d_head),  # thesis + antithesis + current
            nn.ReLU(),
            nn.Linear(self.d_head, self.d_head)
        )
        
        # Gating mechanism for update magnitude
        self.update_gate = nn.Linear(self.d_head * 2, 1)  # current + proposed_update
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # Initialize opposing projections to encourage initial opposition
        nn.init.orthogonal_(self.thesis_proj.weight)
        nn.init.orthogonal_(self.antithesis_proj.weight)
        # Make antithesis start as negative of thesis for initial opposition
        with torch.no_grad():
            self.antithesis_proj.weight.data = -self.thesis_proj.weight.data
    
    def compute_attention_weights(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Standard scaled dot-product attention weights"""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        return F.softmax(scores, dim=-1)
    
    def extract_opposing_summaries(
        self, 
        attn_weights: torch.Tensor, 
        values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract thesis and antithesis summaries using learned opposing transformations
        
        Args:
            attn_weights: [batch, seq_len, seq_len] attention weights
            values: [batch, seq_len, d_head] value vectors
            
        Returns:
            thesis: [batch, seq_len, d_head] thesis summary for each position
            antithesis: [batch, seq_len, d_head] antithesis summary for each position
        """
        # Standard attention-weighted sum
        context = torch.matmul(attn_weights, values)  # [batch, seq_len, d_head]
        
        # Apply opposing transformations
        thesis = self.thesis_proj(context)
        antithesis = self.antithesis_proj(context)
        
        return thesis, antithesis
    
    def measure_tension(self, thesis: torch.Tensor, antithesis: torch.Tensor) -> torch.Tensor:
        """
        Measure how much the thesis and antithesis disagree
        
        Args:
            thesis: [batch, seq_len, d_head]
            antithesis: [batch, seq_len, d_head]
            
        Returns:
            tension: [batch, seq_len, 1] tension scores (0 = aligned, 1 = opposed)
        """
        combined = torch.cat([thesis, antithesis], dim=-1)
        tension = self.tension_net(combined)
        return tension
    
    def synthesize_update(
        self, 
        current_state: torch.Tensor,
        thesis: torch.Tensor, 
        antithesis: torch.Tensor
    ) -> torch.Tensor:
        """
        Negotiate a synthesis between opposing views
        
        Args:
            current_state: [batch, seq_len, d_head] current token representations
            thesis: [batch, seq_len, d_head]
            antithesis: [batch, seq_len, d_head]
            
        Returns:
            update: [batch, seq_len, d_head] proposed update to current state
        """
        combined = torch.cat([thesis, antithesis, current_state], dim=-1)
        synthesis = self.synthesis_net(combined)
        
        # Gate the update magnitude
        gate_input = torch.cat([current_state, synthesis], dim=-1)
        gate = torch.sigmoid(self.update_gate(gate_input))
        
        # Small update step
        update = gate * (synthesis - current_state) * 0.1  # Small learning rate
        return update
    
    def check_stability(
        self, 
        old_state: torch.Tensor, 
        new_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Check if tokens have stabilized (small change between iterations)
        
        Args:
            old_state: [batch, seq_len, d_head]
            new_state: [batch, seq_len, d_head]
            
        Returns:
            stable: [batch, seq_len] boolean mask indicating stable tokens
        """
        change_magnitude = torch.norm(new_state - old_state, dim=-1)
        stable = change_magnitude < self.stability_threshold
        return stable
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with dialectical attention
        
        Args:
            x: [batch, seq_len, d_model] input tokens
            mask: [batch, seq_len, seq_len] attention mask
            return_diagnostics: if True, return debugging information
            
        Returns:
            output: [batch, seq_len, d_head] refined representations
            diagnostics: dict with debugging info (if return_diagnostics=True)
        """
        batch_size, seq_len, _ = x.shape
        
        # Standard attention setup
        q = self.q_proj(x)  # [batch, seq_len, d_head]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Compute attention weights once and reuse
        attn_weights = self.compute_attention_weights(q, k)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_weights = self.dropout(attn_weights)
        
        # Initialize current state with standard attention
        current_state = torch.matmul(attn_weights, v)
        
        # Dialectical refinement loop
        diagnostics = {
            'rounds_per_token': torch.zeros(batch_size, seq_len),
            'final_tensions': torch.zeros(batch_size, seq_len),
            'thesis_antithesis_similarity': [],
            'update_magnitudes': []
        } if return_diagnostics else None
        
        # Track which tokens are still active (not yet stable)
        active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        for round_idx in range(self.max_rounds):
            if not active_mask.any():
                break
                
            # Extract opposing summaries
            thesis, antithesis = self.extract_opposing_summaries(attn_weights, v)
            
            # Measure tension
            tension = self.measure_tension(thesis, antithesis).squeeze(-1)  # [batch, seq_len]
            
            # Only update active (unstable) tokens
            active_thesis = thesis * active_mask.unsqueeze(-1)
            active_antithesis = antithesis * active_mask.unsqueeze(-1)
            active_current = current_state * active_mask.unsqueeze(-1)
            
            # Synthesize updates
            update = self.synthesize_update(active_current, active_thesis, active_antithesis)
            new_state = current_state + update
            
            # Check stability
            stable_this_round = self.check_stability(current_state, new_state)
            
            # Update diagnostics
            if return_diagnostics:
                diagnostics['rounds_per_token'][active_mask] = round_idx + 1
                diagnostics['final_tensions'] = tension
                
                # Similarity between thesis and antithesis
                cos_sim = F.cosine_similarity(thesis, antithesis, dim=-1)
                diagnostics['thesis_antithesis_similarity'].append(cos_sim.mean().item())
                
                # Update magnitudes
                update_mag = torch.norm(update, dim=-1)
                diagnostics['update_magnitudes'].append(update_mag.mean().item())
            
            # Update states and active mask
            current_state = new_state
            active_mask = active_mask & ~stable_this_round
        
        if return_diagnostics:
            return current_state, diagnostics
        else:
            return current_state


class MultiHeadDialecticalAttention(nn.Module):
    """Multi-head version of dialectical attention"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        max_rounds: int = 3,
        stability_threshold: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Create multiple dialectical attention heads
        self.heads = nn.ModuleList([
            DialecticalAttentionHead(
                d_model=d_model,
                d_head=self.d_head,
                max_rounds=max_rounds,
                stability_threshold=stability_threshold,
                dropout=dropout
            ) for _ in range(num_heads)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len, seq_len] 
            return_diagnostics: return debugging info
            
        Returns:
            output: [batch, seq_len, d_model]
            diagnostics: dict (if return_diagnostics=True)
        """
        head_outputs = []
        all_diagnostics = [] if return_diagnostics else None
        
        for head in self.heads:
            if return_diagnostics:
                head_out, head_diag = head(x, mask, return_diagnostics=True)
                head_outputs.append(head_out)
                all_diagnostics.append(head_diag)
            else:
                head_out = head(x, mask, return_diagnostics=False)
                head_outputs.append(head_out)
        
        # Concatenate heads
        concat_output = torch.cat(head_outputs, dim=-1)  # [batch, seq_len, d_model]
        
        # Final projection
        output = self.out_proj(concat_output)
        output = self.dropout(output)
        
        if return_diagnostics:
            # Aggregate diagnostics across heads
            aggregated_diagnostics = {
                'per_head_diagnostics': all_diagnostics,
                'avg_rounds_per_token': torch.stack([d['rounds_per_token'] for d in all_diagnostics]).mean(dim=0),
                'avg_final_tensions': torch.stack([d['final_tensions'] for d in all_diagnostics]).mean(dim=0)
            }
            return output, aggregated_diagnostics
        
        return output


# Example usage and testing
if __name__ == "__main__":
    # Test the dialectical attention head
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create attention layer
    dialectical_attn = MultiHeadDialecticalAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_rounds=3,
        stability_threshold=0.1
    )
    
    # Forward pass with diagnostics
    output, diagnostics = dialectical_attn(x, return_diagnostics=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Average rounds per token: {diagnostics['avg_rounds_per_token'].mean():.2f}")
    print(f"Average final tension: {diagnostics['avg_final_tensions'].mean():.3f}")
    
    # Show head-specific diagnostics
    for i, head_diag in enumerate(diagnostics['per_head_diagnostics']):
        print(f"\nHead {i}:")
        print(f"  Thesis-antithesis similarity progression: {head_diag['thesis_antithesis_similarity']}")
        print(f"  Update magnitude progression: {head_diag['update_magnitudes']}")
