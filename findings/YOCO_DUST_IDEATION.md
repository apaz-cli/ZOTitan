# YOCO x Dust ideation -> moved

Consolidated (with the rest of the swarm) in `~/git/clean-site/blog/_YOCO_Dust_Ideas.md`.
Underscore prefix keeps it out of `generate.py`'s index.

Key corrections after the swarm pass, so they are not lost if someone reads only this file:

1. The handoff is TWO channels: `X^{L/2}` enters the cross-decoder as the residual stream as well as
   being projected to `K^,V^`. So the credit hub is `X^{L/2}`, not the cache. The residual channel is
   diagonal (Dust-exact); only the cache channel needs the analytic pullback.
2. SNR per FLOP, not SNR, is the right currency: probing an `O(T^2)` layer buys `T` rewards for
   `T^2` work, so its virtual-population advantage decays as `1/T`. Windowed layers are flat in `T`.
3. A V-only cache pullback is not a valid test; the `K` path's softmax-centering term vanishes only
   for one-hot attention, which is the regime where `gamma=0` was already fine.
4. Measure bias-of-the-average (`future_over_own`, `weight_cos` in `/home/apaz/git/dust/leakage/`),
   not single-draw cosine. And at random init, where `A_tt ~ 1/t` makes the bias maximal: no training
   run needed to falsify.
