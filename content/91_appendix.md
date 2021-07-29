## Experiment Hyper-Parameters and Overview {#sec:appendixhyper}

| Experiment                  | Entities                                | Dynamics                   | Walls   | Batch size | Environment steps per training step | Max Training Steps |
| --------------------------- | --------------------------------------- | -------------------------- | ------- | ---------- | ----------------------------------- | ------------------ |
| Rendezvous                  | 20 agents                               | Double-Integrator Unicycle | Bounded | 1000       | 164000                              | 160                |
| Single-Evader Pursuit       | 10 agents, 1 evader                     | Single-Integrator Unicycle | Torus   | 10200      | 102000                              | 500                |
| Multi-Evader Pursuit        | 50 agents, 5 evaders (respawning)       | Single-Integrator Unicycle | Torus   | 10200      | 102000                              | 500                |
| Box Assembly                | 10 agents, 4 boxes                      | Single-Integrator Unicycle | Bounded | 5000       | 250000                              | 200                |
| Box Clustering (2 clusters) | 10 agents, 2 clusters with 2 boxes each | Single-Integrator Unicycle | Bounded | 10000      | 512000                              | 2000               |
| Box Clustering (3 clusters) | 20 agents, 3 clusters with 4 boxes each | Single-Integrator Unicycle | Bounded | 100000     | 12800000                            | 500                |
