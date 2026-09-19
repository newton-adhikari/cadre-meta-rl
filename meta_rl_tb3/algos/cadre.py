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

"""
