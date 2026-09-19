"""CADRE: Context-Adaptive Differential-drive Robot Environment meta-RL.

CADRE combines two adaptation mechanisms:

    1. Context adaptation
       z_t = f_φ(τ_{t-K:t-1})
       Online context inference from the last K completed transitions.
       The encoder f_φ (a GRU) is trained end-to-end with the policy.

    2. Gradient adaptation
       θ' = θ - α ∇_θ L(D_support; z)
       One (or more) inner-loop FOMAML gradient steps applied to the
       support data collected with the current context.

The combination is the primary comparison point against:
  - FOMAML alone (no context encoder): CADRE with encoder_type="none"
  - Context-only (no gradient step):   CADRE with num_inner_steps=0
  - PPO fine-tuning baseline:          train_ppo_mt.py

Causal context update protocol
---------------------------------
At time step t:
  1. Compute z_t = encoder(buffer[-K:])   ← uses transitions up to t-1
  2. Select action a_t using π_θ(· | s_t, z_t)
  3. Execute a_t, observe r_t, s_{t+1}
  4. Append (s_t, a_t, r_t, s_{t+1}) to buffer
  5. z_{t+1} = encoder(buffer[-(K):])     ← will use t-th transition

The encoder never sees the transition that resulted from the action it
conditioned — the update is strictly causal.

Meta-gradient correctness
--------------------------
Same in-process pattern as the corrected MAML:
  - ``param.clone()`` retains graph connection to meta-parameters.
  - Rollouts collected under ``torch.no_grad()``.
  - Loss re-evaluated with gradients enabled on the stored (obs, action) pairs.
  - FOMAML: inner gradient detached (``grad.detach()``), not the params.
  - The encoder parameters φ are part of the meta-update graph.

"""
