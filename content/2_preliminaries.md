# Preliminaries {#sec:preliminaries}

In this chapter, we introduce the preliminaries we need for our work. We
describe reinforcement learning and two specific policy gradient training
methods used for deep reinforcement learning in general and then the specifics
of RL for multi-agent systems. While introducing the different variants and
properties of multi-agent reinforcement learning we also describe the related
work.

## Reinforcement learning

Reinforcement learning is a method of training an agent to solve a task in an
environment. As opposed to supervised learning, there is no explicit label /
path to a solution. Instead, the agent iteratively takes actions that affect its
environment, and receives a reward when specific conditions are met. The reward
can be dense (positive or negative signal at every step) or very sparse (only a
binary reward at the end of an episode).

Reinforcement learning problems are usually defined as Markov Decision Processes
(MDPs). An MDP is an extension of a Markov chain, which is a set of states with
a transition probability between each pair of states. The probability only
depends on the current state, not the previous states.

MDPs extend Markov chains, adding a set of actions (that allow an agent to
influence the transition to the next state) and rewards (that motivate the
agent). An MDP thus consists of:

- A set of states $S$ (the _state space_)
- A set of actions $A$ (the _action space_)
- The transition probability that a specific action in a specific state leads to
  specific second state $T(s,a,s') = P(s'|s,a)$
- The reward function that specifies the immediate reward an agent receives
  depending on the previous state, the action, and the new state
  $R: S × A × S → \mathbb{R}$

The MDP can either run for an infinite period or finish after a number of
transitions (the transition function then always returns the same state).

The method by which an agent chooses its actions based on the current state is
called a _policy_. The policy defines a probability for taking each action based
on a specific state.

Based on a policy we can also define the _value function_, which is the sum of
all future rewards that an agent gets based on an initial state and a specific
policy. The value function can also include an exponential decay for weighing
future rewards, which is especially useful if the horizon is infinite.

Often we need to extend MDPs to allow for an observation model: A _partially
observable Markov decision process_ (POMDP) is an extension to MDPs that
introduces an indirection between the set of states and the input to the policy
(called an _observation_). In the definition of a POMDP a set of observations is
added with a probability of each state leading to a specific observation. The
policy now depends on the observation and the action instead of the state and
the action. An observation does not necessarily include all the information from
the corresponding state.

To successfully solve a reinforcement learning task, we need to find a policy
that has a high expected reward — we want to find the _optimal policy function_
that has the highest value function on the initial states of our environment.
Since finding the optimal policy directly is impossible for even slightly
complicated tasks, we instead use optimization techniques.

## Actor-critic training methods

Policy gradient training methods are a reinforcement learning technique that
optimize the parameters of a policy using gradient descent. Actor-critic methods
optimize both the policy and an estimation of the value function at the same
time.

There are multiple commonly used actor critic training methods such as Trust
Region Policy Optimization (TRPO)
[@{http://proceedings.mlr.press/v37/schulman15.html}], Proximal Policy
Optimization (PPO) [@ppo] and Soft Actor Critic
[@{https://arxiv.org/abs/1801.01290}]. We run most of our experiments with one
of the most commonly used methods (PPO) and also explore a new method that
promises stabler training (PG-TRL [@trl]).

### Proximal Policy Optimization (PPO)

[@ppo]: https://arxiv.org/abs/1707.06347

Proximal Policy Optimization [@ppo] is an actor-critic policy gradient method
for reinforcement learning. In PPO, each training step consists of collecting a
set of trajectory rollouts, then optimizing a "surrogate" objective function
using stochastic gradient descent. The surrogate objective function approximates
the policy gradient while enforcing a trust region by clipping the update steps.
PPO is a successor to Trust Region Policy Optimization (TRPO) with a simpler
implementation and empirically better performance.

PPO optimizes the policy using

$$θ_{k+1} = \text{argmax}_{θ} E_{s,a \sim π_{θ_k}} [L(s, a, θ_k, θ)]$$

Where $π_{θ_k}$ is a policy with parameters $θ$ in training step $k$, $s$ is the
state, $a\sim π_{θ_k}$ is the action distribution according to the policy at
step $k$. L is given by

$$L(s,a,θ_k,θ) = \min \left( \frac{π_θ(a|s)}{π_{θ_k}(a|s)} A^{π_{θ_k}}(s,a), \text{clip}(\frac{π_θ(a|s)}{π_{θ_k}(a|s)}, 1 - ε, 1 + ε) A^{π_{θ_k}}(s,a) \right).$$

$A$ is the advantage of taking a specific action in a specific state as opposed
to the other actions as weighted by the current policy, estimated using
Generalized Advantage Estimation [@{https://arxiv.org/abs/1506.02438}] based on
the estimated value function.

Since the performance of PPO depends on a number of implementation details and
quirks, we use the stable-baselines3 [@stable-baselines3] implementation of
PPO-Clip, which has been shown to perform as well as the original OpenAI
implementation
[@{https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#results}].

We use PPO as the default for most of our experiments since it is widely used in
other deep reinforcement learning work.

### Trust Region Layers (PG-TRL [@trl]) {#sec:trl}

Differentiable trust region layers are an alternative method to enforce a trust
region during policy updates introduced by @trl. PPO uses a fixed clipping ratio
to enforce the trust region, which can result in unstable training. Instead of
using a fixed clipping boundary, in PG-TRL a surrogate policy is created with a
new layer at the end that projects the unclipped policy back into the trust
region. The trust region and the projection is based on either the KL-divergence
[@doi:10.1214/aoms/1177729694] or the Wasserstein $W_2$ distance
[@{https://www.springer.com/gp/book/9783540710493}].

After each training step, the projected new policy depends on the previous
policy. To prevent an infinite stacking of old policies, the explicitly
projected policy is only used as a surrogate, while the real policy is based on
a learned approximation. To prevent the real policy and the projected policy
from diverging, the divergence between them is computed again in the form of the
KL divergence or the Wasserstein $W_2$ distance. The computed divergence (_trust
region regression loss_) is then added to the policy gradient loss function with
a high factor.

We explore PG-TRL as an alternative training method to PPO.

## Multi-agent reinforcement learning (MARL)

Usually, reinforcement learning environments are modeled as POMDPs. POMDPs only
have a single agent interacting with the environment, so for multi-agent
settings, we need a different system that can model multiple agents interacting
with the world. There are different ways of extending POMDPs to work for
multi-agent tasks. Decentralized POMDPs (Dec-POMDP)
[@{https://www.springer.com/gp/book/9783319289274}] are a generalization of
POMDPs that split the action and observation spaces into those of each agent.

A Dec-POMDP consists of:

- A set of $n$ agents $D=\{1,\dots,n\}$
- A set of states $S$
- A set of joint actions $A$
- A set of transition probabilities between states $T(s,a,s') = P(s'|s,a)$
- A set of joint observations $O$
- An immediate reward function $R: S × A → \mathbb{R}$

Note that this definition is very similar to a normal POMDP. The difference is
that the set of joint observations and joint actions consists of a tuple of the
observations/actions of each individual agent (i.e.
$a\in A = \langle a_1,\dots,a_n \rangle$, where $a_i$ is the action of agent
$i$).

In Dec-POMDPs every agent can have differing observations and actions. The
subset of Dec-POMDPs where the agents are homogenous has been described as a
SwarMDP by @{https://dl.acm.org/doi/10.5555/3091125.3091320}. The joint
observation and action spaces thus become $O = \hat{O^n}$ and $A = \hat{A^n}$,
where $\hat{O}$ and $\hat{A}$ are the observation and action space of each
agent.

## Environment model and learning process

In our experiments, we impose a set of restrictions on the environments and
learning process. The restrictions we impose are mostly based on [@maxpaper].
Here, we describe the major differing factors of both the learning process and
the environments, as well as the variants we choose to consider.

In general, the agents in a multi-agent environment can differ in their
intrinsic properties. For example, they can have different control dynamics,
maximum speeds, different observation systems, or different possible actions. We
only consider environments with homogenous agents: All agents have the same
physical properties, observation space, and action space. They only differ in
their extrinsic properties, e.g., their current position, rotation, and speed.
This also causes them to have a different perspective, different observations
and thus different actions, resulting in differing behavior even when they are
acting according to the same policy. We thus only environments where the agents
are homogenous — each agent has the same possible observations and actions.

We only consider cooperative environments, and we use the same reward function
for all agents. Real-world multi-agent tasks are usually cooperative since in
adversarial environments, where different agents have different goals, one
entity would not have control over multiple adversarial parties.

We focus on global visibility since the additional noise introduced by local
observability would be detrimental to the quality of our results.

For training, we use the centralized-learning/decentralized-execution (CLDE)
approach — a shared common policy is learned for all agents, but the policy is
executed by each agent separately.

PPO and other policy gradient methods are designed for single-agent
environments. We adapt PPO to the multi-agent setting without making any major
changes: The policy parameters are shared for all agents, and each agent gets
the same global reward. The value function for advantage estimation is based on
the same architecture as the policy. Since the stable-baselines3 implementation
of PPO is already written for vectorized environments (collecting trajectories
from many environments running in parallel), we create a new VecEnv
implementation that flattens multiple agents in multiple environments.

Similarly to the setup used for TRPO by @maxpaper, we collect the data of each
agent as if that agent was the only agent in the world. For example, a batch of
a single step of 10 agents each in 20 different environment becomes 200 separate
training samples. Each agent still only has access to its own local
observations, not the global system state. This means that during inference
time, each agent has to act independently, based on the observations it makes
locally.

## Aggregation methods

The observations of each agent in an MARL task contains a varying number of
observables. The observables can be clustered into groups where each observable
is of the same kind and shape. For example, one observable group would contain
all the neighboring agents, while another would contain, e.g., a specific type
of other observed objects in the world.

We need a method to aggregate the observations in one observable group into a
format that can be used as the input of a neural network. Specifically, this
means that the output of the aggregation method needs to have a fixed
dimensionality. In the following, we present different aggregation methods and
their properties.

### Concatenation

The simplest aggregation method is concatenation, where each observable is
concatenated along the feature dimension into a single observation vector. This
method has a few issues however: We can have a varying number of observables,
but since the input size of the neural network is fixed, we need to set a
maximum number of observables, and set the input dimensionality of the policy
architecture proportional to that maximum. Concatenation also ignores the other
useful properties of the observables, namely the uniformity (the feature at
index $i$ of one observable has the same meaning as the feature at index $i$ of
every other observable in a group) and the exchangeability (the order of the
observables is either meaningless or variable).

Concatenation scales poorly with a large number of observables since the input
size of the neural network has to scale proportionally to the maximum number of
observables. It is used for example by @mpe to aggregate the neighboring agents'
observations and actions.

### Mean aggregation

Instead of concatenating each element $o_i$ in an observable group $O$, we can
also interpret each element as a sample of a distribution that describes the
current system state. We use the empirical mean of the samples to retrieve a
representation of the system state $ψ_O$ based on all observed observables, as
shown in @eq:meanagg.

$$ψ_O = μ_O = \frac{1}{|O|} \sum_{o_i ∈ O} o_i$$ {#eq:meanagg}

The encoder is an arbitrary function that maps the observation into a latent
space, and can be represented by a neural network with shared weights across the
observables in an observable group. @maxpaper used mean aggregation for deep
multi-agent reinforcement learning and compared it to other aggregation methods.
@gregor applied mean aggregation to more complex tasks in more realistic
simulated environments.

Mean aggregation is strongly related to mean field theory. Mean field theory is
a general principle of modeling the effect that many particles have by averaging
them into a single field, ignoring the individual variances of each particle.
The application of mean field theory for multi-agent systems were formally
defined by @{https://link.springer.com/article/10.1007/s11537-007-0657-8} as
_Mean Field Games_. In MARL, mean field Q-learning and mean field actor-critic
was defined and evaluated by @{http://proceedings.mlr.press/v80/yang18d.html}.
@meanfielduav use mean fields productively for control of numerous unmanned
aerial vehicles.

[@gregor]:
  https://www.semanticscholar.org/paper/Using-M-Embeddings-to-Learn-Control-Strategies-for-Gebhardt-H%C3%BCttenrauch/9f550815f8858e7c4c8aef23665fa5817884f1b3
[@meanfielduav]: https://ieeexplore.ieee.org/abstract/document/9137257

### Aggregation with other pooling functions

Instead of using the empirical mean, other aggregation methods can also be used:

Max pooling:

$$ψ_O = \max_{o_i ∈ O} o_i$$

Softmax pooling:

$$ψ_O = \sum_{o_i ∈ O} o_i \frac{e^{o_i}}{\sum_{o_j ∈ O} e^{o_j}}$$

Max-pooling is widely used in convolutional neural networks to reduce the image
dimensionality where it consistently outperforms mean (average) pooling. Softmax
aggregation was used by @{https://arxiv.org/abs/1703.04908} for MARL.

### Bayesian aggregation {#sec:bayesianagg1}

Aggregation with Gaussian conditioning works by starting from a Gaussian prior
distribution and updating it using a probabilistic observation model for every
seen observation. Gaussian conditioning is widely used in applications such as
Gaussian process regression
[@{https://link.springer.com/chapter/10.1007/978-3-540-28650-9_4}].

We consider Bayesian aggregation as defined by [@bayesiancontextaggregation, sec
7.1]. We introduce $z$ as a random variable with a Gaussian distribution:

$$z \sim \mathcal{N}(μ_z,σ_z^2)\quad (p(z) ≡ \mathcal{N}(μ_z,σ_z^2))$$

Initially, this random variable is estimated using a diagonal Gaussian prior as
an a-priori estimate:

$$p_0(z)≡\mathcal{N}(μ_{z_0}, diag(σ_{z_0}^2))$$

This prior is then updated with Bayesian conditioning using each of the observed
elements $r_i$. We interpret each observation as a new sample from the
distribution $p(z)$, each with a mean $r_i$ and a standard deviation $σ_{r_i}$.
We use the probabilistic observation model and consider the conditional
probability $$p(r_i|z) ≡ \mathcal{N}(r_n|z, σ_{r_i}^2).$$

With standard Gaussian conditioning
[@{https://www.springer.com/gp/book/9780387310732}] this leads to the closed
form factorized posterior description of $z$:

$$\sigma_z^2 = \frac{1}{\frac{1}{\sigma_{z_0}^2} + \sum_{i=1}^n{\frac{1}{\sigma_{r_i}^2}}}$$

$$\mu_z = \mu_{z_0} + \sigma_z^2 \cdot \sum_{i=1}^{n}{\frac{(r_i-\mu_{z_0})}{\sigma_{r_i}^2}}$$

Bayesian aggregation was used by @bayesiancontextaggregation for context
aggregation in conditional latent variable (CLV) models. There is no related
work using Bayesian aggregation for MARL.

### Attention mechanisms {#sec:mha}

Attention mechanisms are commonly used in other areas of machine learning.

An attention function computes some compatibility between a _query_ $Q$ and a
_key_ $K$, and uses this compatibility as the weight to compute a weighted sum
of the _values_ $V$. The query, key, and value are all vectors.

[@att]:
  https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

<!-- For attentive aggregation, the observations from the different agents are first
attended to by some attention mechanism, then the resulting new feature vector
is aggregated using some aggregation mechanism. -->

<!-- {attention: 30,31,32,33 (from ARE paper)} -->

The multi-head attention mechanism was introduced by @att and is the core of the
state-of-the-art model for many natural language processing tasks.

We only consider the specific multi-head residual masked self attention variant
of attention mechanisms for observation aggregation used by @openai.

There, the attention function in is a scaled dot-product attention as described
by [@att]:

$$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

$d_k$ is the dimensionality of the key $K$.

Instead of using only a single attention function, [@att] uses multiple
independent attention heads. The inputs ($Q, K, V$) of each of the heads
$1,\dots,n$ as well as the concatenated output is transformed with a separately
learned dense layers (described as weight matrices $W_i^Q, W_i^K, W_i^V, W^O$).
The full multi-head attention module $\text{MHA}()$ thus looks like this:

$$\text{head}_i = \text{Attention}(QW_i^Q,KW_i^K,VW_i^V)$$

$$\text{MHA}(Q,K,V)=\text{concat}(\text{head}_1,\dots,\text{head}_n)W^O$$

For self-attention, the query $Q$, the key $K$ and the value $V$ are all set to
the same input value. In [@openai], the authors combine the multi-head attention
with a mean aggregation. Note that they only use the attention mechanism to
individually transform the information from each separate observable into a new
feature set instead of directly using it as a weighing function for the
aggregation.

@{http://proceedings.mlr.press/v97/iqbal19a.html} use a different approach with
a decentralized policy and a centralized critic. An attention mechanism is used
to weigh the features from the other agents, then aggregates them to calculate
the value function.

@are first encode all aggregatable observations together with the proprioceptive
observations with an "extrinsic encoder", then compute attention weights using
another encoder from that and aggregate the features retrieved from a separate
"intrinsic encoder" using those weights.

[@are]: https://ieeexplore.ieee.org/document/9049415

### Other aggregation methods

@maxpaper also did experiments with aggregation into histograms and aggregation
with radial basis functions, though the results indicated they were outperformed
by a neural network encoder with mean aggregation.

<!-- @tocommunicate and
- based on temporal information (recurrent nns)
-

-->
