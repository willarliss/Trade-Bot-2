# Trade-Bot-2

This project tests a continuous stock trading environment for reinforcement learning.

 - `TradingEnv9` is main environment
 - `TD3_TradingEnv9_main_42` parameters:
   - Initialization: `buffer_size` = 1,000,000; `batch_size` = 100; `gamma` = 0.99; `tau` = 0.00001; `policy_freq` = 2; `lr` = 0.001; `policy_noise` = 0.2; `noise_clip` = 0.5; `expl_noise` = 0.15; `starting_step` = 30,000; `init_thresh` = 2.0;
   - Robust: . ; `lr` = 0.0001; `alpha` = 0.01; `starting_step` = 15,000;
 - `TD3_TradingEnv9_main_86` parameters:
   - Initialization: `buffer_size` = 1,000,000; `batch_size` = 100; `gamma` = 0.99; `tau` = 0.00001; `policy_freq` = 2; `lr` = 0.001; `policy_noise` = 0.2; `noise_clip` = 0.5; `expl_noise` = 0.15; `starting_step` = 25,000; `init_thresh` = 2.0;
   - Robust: . ; `lr` = 0.0005; `alpha` = 0.001; `starting_step` = 10,000;
 
   