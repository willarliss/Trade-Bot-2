# Trade-Bot-2

This project tests a continuous stock trading environment for reinforcement learning.

 - Use `TradingEnv9`
 - Base parameters:
   - `buffer_size` = 1,000,000; `batch_size` = 100; `gamma` = 0.99; `tau` = 0.00001; `policy_freq` = 2; `lr` = 0.001; `policy_noise` = 0.2; `noise_clip` = 0.5; `expl_noise` = 0.15; `starting_step` = 20,000;
 - `TD3_TradingEnv9_main_42` parameters:
   - Initialization:  . ; `init_thresh` = 2.0; `starting_step` = 30,000;
   - Robust:  . ; `lr` = 0.0001; `reg` = 0.01; `starting_step` = 15,000;
 - `TD3_TradingEnv9_main_86` parameters:
   - Initialization:  . ; `init_thresh` = 2.0;
   - Robust:  . ; `lr` = 0.0005; `reg` = 0.001; `starting_step` = 10,000;
 - `TD3_TradingEnv9_main_33` parameters:
   - Initialization:  . ; `init_thresh` = 5.0;
   - Robust:  . ; `reg` = 0.0001; `starting_step` = 10,000;
 
   
    