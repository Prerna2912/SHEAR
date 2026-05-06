# SHEAR
SE(3)-Equivariant Conditional Flow Matching for sub-grid scale turbulence closure in Large Eddy Simulation. Trains a generative model over SGS stress tensors using e3nn and torchdiffeq, conditioned on filtered velocity gradients from JHTDB. Benchmarked against Dynamic Smagorinsky, WALE, and neural baselines on a-priori metrics.
