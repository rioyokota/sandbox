vocab_size = 128000              # vocabulary size
hidden_dim = 7168                # transformer hidden dimension
num_heads = 128                  # number of attention heads
latent_head_dim = 64             # per-head latent dimension after compression (decoupled Q/K)
q_comp_dim = 1536                # query compression dimension
kv_comp_dim = 512                # key/value compression dimension

# Embedding & output matrices (not tied)
embedding_params = vocab_size * hidden_dim            # embedding matrix
output_params = hidden_dim * vocab_size               # output projection matrix (decoder logits)

# Attention (MLA) parameters per layer
Wq_down = hidden_dim * q_comp_dim                     # down-projection for queries
Wq_up   = q_comp_dim * (num_heads * latent_head_dim)  # up-projection for queries
Wkv_down = hidden_dim * kv_comp_dim                   # shared down-projection for keys/values
Wk_up   = kv_comp_dim * (num_heads * latent_head_dim) # up-projection for keys
Wv_up   = kv_comp_dim * (num_heads * latent_head_dim) # up-projection for values
W_out   = (num_heads * latent_head_dim) * hidden_dim  # output projection of attention
attn_params_per_layer = Wq_down + Wq_up + Wkv_down + Wk_up + Wv_up + W_out

# Total attention params for all 61 layers
num_layers = 61
attn_total_params = attn_params_per_layer * num_layers

# Mixture-of-Experts (MoE) Feed-Forward parameters
moe_layers = 58                                       # number of MoE layers (61 layers minus 3 dense)
experts_per_layer = 256                               # routed experts per MoE layer
shared_experts_per_layer = 1                          # shared expert per MoE layer
ffn_intermediate_dim = 2048                           # expert intermediate hidden dim
ffn_factor = 3                                        # use 3 for gated FFN (e.g. SwiGLU), 2 for standard ReLU

# Parameters per expert (two linear layers, with GLU gating if ffn_factor=3)
expert_params = ffn_factor * hidden_dim * ffn_intermediate_dim

# MoE layer total (256 routed + 1 shared expert)
moe_params_per_layer = (experts_per_layer + shared_experts_per_layer) * expert_params

# Total MoE expert params across all MoE layers
moe_total_params = moe_params_per_layer * moe_layers

# Gating network parameters (one linear layer [hidden_dim -> 256] per MoE layer)
gating_params_per_layer = hidden_dim * experts_per_layer
gating_total_params = gating_params_per_layer * moe_layers

# Dense FFN layers (first 3 layers use a standard FFN)
dense_layers = 3
dense_ffn_intermediate_dim = 4 * hidden_dim         # assume 4x hidden dim for dense FFN
dense_ffn_factor = ffn_factor                       # (if using same activation function)
dense_ffn_params_per_layer = dense_ffn_factor * hidden_dim * dense_ffn_intermediate_dim
dense_ffn_total_params = dense_ffn_params_per_layer * dense_layers

# Normalization and scaling parameters (RMSNorm in each layer, etc.)
norm_params = num_layers * hidden_dim               # one RMSNorm gamma per layer (approx)

# Multi-Token Prediction (MTP) module parameters
mtp_unique_params = 11.5e9   # 11.5B unique params

total_params = (embedding_params + output_params + 
                attn_total_params + 
                moe_total_params + gating_total_params + 
                dense_ffn_total_params + 
                norm_params + 
                mtp_unique_params)

# Compute Active Parameters (per token)
active_experts_per_layer = 8    # routed experts used per token
active_shared_experts = 1       # shared expert always active

# Active MoE params per layer = (active experts count * params per expert)
active_moe_params_per_layer = (active_experts_per_layer + active_shared_experts) * expert_params
active_moe_total_params = active_moe_params_per_layer * moe_layers

active_params = (embedding_params + output_params + 
                 attn_total_params + 
                 dense_ffn_total_params + 
                 active_moe_total_params + 
                 gating_total_params + 
                 norm_params)

print(f"Embedding = {embedding_params/1e9:.1f}B")
print(f"Output    = {output_params/1e9:.1f}B")
print(f"Attention = {attn_total_params/1e9:.1f}B")
print(f"Dense     = {dense_ffn_total_params/1e9:.1f}B")
print(f"Gating    = {gating_total_params/1e9:.1f}B")
print(f"MoE       = {moe_total_params/1e9:.1f}B")
print(f"MoE (act) = {active_moe_total_params/1e9:.1f}B")
print(f"Norm      = {norm_params/1e9:.1f}B")
print(f"MTP       = {mtp_unique_params/1e9:.1f}B")
print(f"Total     = {total_params/1e9:.1f}B")
print(f"Active    = {active_params/1e9:.1f}B")
