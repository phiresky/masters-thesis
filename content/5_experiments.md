# Experiments {#sec:experiments}

## Experimental setup

All of our experiments were run using stable-baselines3 (sb3). _OpenAI
Baselines_ is a deep reinforcement learning framework that provides
implementations of various reinforcement learning algorithms for TensorFlow.
Stable-baselines is an extension of the original code base with better
documentation and more flexibility for custom architectures and environments.
sb3 is the continuation of the stable-baselines project, rewritten using
PyTorch.

We extended sb3 for the multi-agent use case by adapting the vectorized
environment implementation to support multiple agents in a single environment,
as well as adapting the training and evaluation functionality to correctly
handle the centralized-learning decentralized-execution method. We also
implement Trust Region Layers [@trl] as a new training algorithm for sb3 in
order to be able to directly compare PPO and TRL.

We optimized the hyper parameters using Optuna
[@{https://dl.acm.org/doi/10.1145/3292500.3330701}].

{describe batch size, other meta settings}

## Considered tasks

To evaluate the different aggregation methods we need simulated environments
where multiple agents can cooperatively solve a task. Since most of the commonly
used tasks used to benchmark reinforcement learning algorithms are designed for
a single agent, we use custom built environments.

The following shows which specific tasks we consider.

### Rendezvous task {#sec:rendezvous}

In the rendezvous task, a set of $n$ agents try to meet up in a single point. An
example is shown in @fig:rendezvous1. Our implementation of the rendezvous task
is modeled after [@maxpaper].

![Visualization of one successful episode of the rendezvous task (from [@maxpaper])](images/rendezvous1.png){#fig:rendezvous1}

The agents are modeled as infinitesimal dots without collisions. They use
double-integrator unicycle dynamics
[@{https://ieeexplore.ieee.org/document/976029}], so the action outputs are the
acceleration of the linear velocity ($\dot{v}$) and the angular velocity
($\dot{ω}$). The agents move around in a square world that either has walls at
the edge or is on a torus (the position is calculated modulo $100$).

The observation space of agent $a_i$ comprises the following information:

1.  The own linear velocity $v_{a_i}$
2.  The own angular velocity $ω_{a_i}$
3.  The ratio of currently visible agents
4.  If the observation space is not a torus: the distance and angle to the
    nearest wall
5.  For each neighboring agent:
    1.  The distance between the current agent and the neighbor
    2.  The cos and sin of the relative bearing (the angle between the direction
        the agent is going and the position of the neighbor)
    3.  The cos and sin of the neighbor's bearing to us
    4.  The velocity of the neighbor relative to the agent's velocity
    5.  The ratio of currently visible agents for the neighbor

The reward that every agent gets is the mean of all the pairwise agent
distances:

$$r = \frac{1}{n} \sum_{i=0}^n \sum_{j=i+1}^n ||p_i-p_j||$$

The episode ends after 1024 time steps.

### Single-Evader Pursuit task

In the pursuit task, multiple pursuers try to catch an evader. The evader agent
has a higher speed than the pursuers, so the pursuers need to cooperate to be
able to catch it. The world is a two-dimensional torus (the position $(x, y)$
are floating point numbers modulo 100). If the world was infinite, the evader
could run away in one direction forever, and if it had walls, the pursuers could
easily corner the evader. The pursuit tasks are modeled after [@maxpaper]. An
example episode of the pursuit task is in [@fig:pursuit1].

![Visualization of one successful episode of the pursuit task (from [@maxpaper]). The pursuers are in blue, the evader is in red.](images/pursuit1.png){#fig:pursuit1}

The agents are modeled with single-integrator unicycle dynamics. The action
outputs are the linear velocity ($v$) and the angular velocity ($ω$).

The evader is not part of the training and uses a hard-coded algorithm based on
maximizing Voronoi-regions as described by
@{https://www.sciencedirect.com/science/article/abs/pii/S0005109816301911}. It
is thus part of the environment and not an "agent" as seen from the perspective
of the training algorithm.

The observation space for the single-evader pursuit task comprises the following
information:

1. The ratio of agents that are visible
2. For each neighboring agent:
   1. The distance between the current agent and the neighbor
   2. The cos and sin of the relative bearing (the angle between the direction
      the agent is going and the position of the neighbor)
   3. The cos and sin of the neighbor's bearing to us
3. For each evader:
   1. The distance between the current agent and the evader
   2. The cos and sin of the relative bearing

The reward function of all the agents is the minimum distance of any agent to
the evader:

$$r = min_{i=0}^n ||p_{a_i} - p_{e}||$$

The episode ends once the evader is caught or 1024 timesteps have passed. The
evader is declared as caught if the distance is minimum distance between an
agent and the evader is less than $1\%$ of the world width.

### Multi-evader pursuit

The multi-evader pursuit task is the same as the normal pursuit task, except
there are multiple hard-coded evaders.

The reward here is different, since defining the reward as for the single-evader
task is not obvious. The reward is +1 whenever an evader is caught, 0 otherwise.
An evader is declared caught if the distance to the nearest pursuer is less than
$\frac{2}{100}$ of the world width. Contrary to the single-evader task, the
episode does not end when an evader is caught and instead always runs for 1024
timesteps.

### Box assembly task

In the box assembly task, the agents are modeled similar to Kilobots, as
described by @gregor. The agents are two-dimensional circles with collision.

For simplicity, we again use single-integrator unicycle dynamics with the linear
velocity $v$ and the angular velocity $ω$ as the action outputs instead of the
specific dynamics of real Kilobots.

The world is a square and contains a few two-dimensional boxes (squares) as
obstacles. For the box assembly task, the goal is to get all the boxes as close
together as possible. Since the agents are much smaller than the boxes, moving a
box is hard for a single agent. The reward is the derivation of the negative sum
of the pairwise distances between the boxes. We run this task with four boxes
and 10 agents.

An example of a successful episode of the task is in @fig:assembly.

![Example successful episode of the box assembly task (from [@gregor])](images/box-assembly.png){#fig:assembly}

[@gregor]:
  https://www.semanticscholar.org/paper/Using-M-Embeddings-to-Learn-Control-Strategies-for-Gebhardt-H%C3%BCttenrauch/9f550815f8858e7c4c8aef23665fa5817884f1b3

The observation space for the box assembly task contains the following
information:

1. The absolute position $(x, y)$ of the current agent
2. The sin, cos of the absolute rotation of the current agent
3. For every neighboring agent:
   1. The distance between the current agent and the neighbor
   2. The sin and cos of the bearing angle to the neighbor
   3. The sin and cos of the neighbor's rotation relative to our rotation
4. For every neighboring box:
   1. The distance between the current agent and the box
   2. The sin and cos of the bearing angle to the box
   3. The sin and cos of the box's rotation relative to our rotation

### Box clustering task

The task setup for box clustering is the same as for the box assembly task,
except that each box is assigned a color. The goal is to move the boxes into an
arrangement such that boxes of the same color are as close together as possible,
while boxes of different colors are far away. Each color of box has an explicit
target one corner of the world. The reward is calculated by taking the sum of
negative distances between each box and its cluster target, then taking the
difference of this after the step and before the step.

The observation space is the same as the observation space of the box assembly
task, except that each cluster of objects is aggregated into a separate
aggregation space, and that there is an additional one-hot encoded cluster ID in
the object observation space.

![Example episode of the clustering task with two clusters.](images/clustering2-episode.drawio.svg){#fig:clustering2}

<!-- ### Other considered tasks

PettingZoo tasks (without results?)

-->
