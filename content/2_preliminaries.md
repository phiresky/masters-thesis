## Preliminaries {#sec:preliminaries}

In this chapter, we introduce the preliminaries we need for our work. We
describe reinforcement learning and two specific policy gradient training
methods used for deep reinforcement learning in general and then the specifics
of RL for multi-agent systems. While introducing the different variants and
properties of multi-agent reinforcement learning we also describe the related
work.

### Reinforcement learning

Reinforcement learning is a method of training an agent to solve a task in an
environment. As opposed to supervised learning, there is no explicit path to a
solution. Instead, the agent iteratively takes actions that affect its
environment, and receives a reward when specific conditions are met. The reward
can be dense (positive or negative signal at every step) or very sparse (only a
binary reward at the end of an episode).

Reinforcement learning problems are usually defined as Markov Decision Processes
(MDPs). An MDP is an extension of a Markov chain, which is a set of states with
a transition probability between each pair of states. The probability only
depends on the current state, not the previous states.

MDPs extend Markov chains, adding a set of actions (that allow an agent to
influence the transition probabilities) and rewards (that motivate the agent).
An MDP thus consists of:

- A set of states (the _state space_)
- A set of actions (the _action space_)
- The transition probability that a specific action in a specific state leads to
  specific second state
- The reward function that the specifies the immediate reward an agent receives
  depending on the previous state, the action, and the new state

The MDP can either run for an infinite period or finish after a number of
transitions.

The method by which an agent chooses its actions based on the current state is
called a _policy_.

Based on a policy we can also define the _value function_, which is the sum of
all future rewards that an agent gets based on an initial state and a specific
policy. The value function can also include an exponential decay for weighing
future rewards.

To successfully solve a reinforcement learning task, we need to find a policy
that has a high expected reward - we want to find the _optimal policy function_
that has the highest value function on the initial states of our environment.
Since finding the optimal policy directly is impossible for even slightly
complicated tasks, we instead use optimization techniques.

### Actor-critic training methods

Policy gradient training methods are a reinforcement learning technique that
optimize the parameters of a policy using gradient descent. Actor-critic methods
also optimize an estimation of the value function at the same time, since

There are multiple commonly used actor critic training methods such as Trust
Region Policy Optimization (TRPO)
[@{http://proceedings.mlr.press/v37/schulman15.html}], Proximal Policy
Optimization (PPO) [@ppo] and Soft Actor Critic
[@{https://arxiv.org/abs/1801.01290}]. We run most of our experiments with of
the most commonly used methods (PPO) and also explore a new method that promises
stabler training (PG-TRL).

#### Proximal Policy Optimization (PPO)

[@ppo]: https://arxiv.org/abs/1707.06347

Proximal Policy Optimization [@ppo] is a actor-critic policy gradient method for
reinforcement learning. In PPO, each training step consists of collecting a set
of trajectory rollouts, then optimizing a "surrogate" objective function using
stochastic gradient descent. The surrogate objective function approximates the
policy gradient while enforcing a trust region by clipping the update steps. PPO
is a successor to Trust Region Policy Optimization (TRPO) with a simpler
implementation and empirically better performance.

Since the performance of PPO depends on a number of implementation details and
quirks, we use the stable-baselines3 [@stable-baselines3] implementation of
PPO-Clip, which has been shown to perform as well as the original OpenAI
implementation
[@{https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#results}].

We use PPO as the default for most of our experiments since it is widely used in
other deep reinforcement learning work.

#### Trust Region Layers (PG-TRL) {#sec:trl}

Differentiable trust region layers are an alternative method to enforce a trust
region during policy updates introduced by @trl. PPO uses a fixed clipping ratio
to enforce the trust region, which can result in unstable training. Instead of
using a fixed clipping boundary, in PG-TRL a surrogate policy is created with a
new layer at the end that projects the unclipped policy back into the trust
region. The trust region and the projection is based on either the KL-divergence
[@doi:10.1214/aoms/1177729694] or the Wasserstein $W_2$ distance
[@{https://www.springer.com/gp/book/9783540710493}].

After each training step, the projected new policy depends on the previous
policy. To prevent an infinite stacking of old policies, the explicitly projected
policy is only used as a surrogate, while the real policy as based on a learned
approximation. To prevent the real policy and the projected policy from
diverging, the divergence between them is computed again in the form of the KL
divergence or the Wasserstein $W_2$. The computed divergence (_trust
region regression loss_) is then added to the policy gradient loss function with
a high factor.

We explore PG-TRL as an alternative training method to PPO.

### Multi-agent reinforcement learning

DecPOMDP, SwarMDP definition and how to solve it using PPO

### Aggregation methods

The observations of each agent in a MARL task contains a varying number of
observables. The observables can be clustered into groups where each observable
is of the same kind and shape. For example, one observable group would contain
all the neighboring agents, while another would contain, e.g., a specific type
of other observed objects in the world.

We need a method to aggregate the observations in one observable group into a
format that can be used as the input of a neural network. Specifically, this
means that the output of the aggregation method needs to have a fixed
dimensionality. In the following, we present different aggregation methods and
their properties.

#### Concatenation

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
observables.

Concatenation is used for example by @mpe to aggregate the neighboring agents'
observations and actions.

#### Mean aggregation

Instead of concatenating each element $o_i$ in an observable group $O$, we can
also interpret each element as a sample of a distribution that describes the
current system state. We encode each of these samples into a latent space that
describes the relevant properties of the system, then use the empirical mean of
the encoded samples to retrieve a representation of the system state $ψ_O$ based
on all observed observables, as shown in @eq:meanagg.

$$ψ_O = μ_O = \frac{1}{|O|} \sum_{o_i ∈ O} \text{encode}(o_i)$$ {#eq:meanagg}

The encoder is an arbitrary function that maps the observation into a latent
space, and can be represented by a neural network with shared weights across the
observables in a observable group. @maxpaper used mean aggregation for deep
multi-agent reinforcement learning and compared it to other aggregation methods.
@gregor applied mean aggregation to more complex tasks in more realistic
simulated environments.

Mean aggregation is strongly related to mean field theory. Mean field theory is
a general principle of modeling the effect that a large number of particles have
by averaging them into a single field, ignoring the individual variances of each
particle. The application of mean field theory for multi-agent systems were
formally defined by
@{https://link.springer.com/article/10.1007/s11537-007-0657-8} as _Mean Field
Games_. In MARL, mean field Q-learning and mean field actor-critic was defined
and evaluated by @{http://proceedings.mlr.press/v80/yang18d.html}. @meanfielduav
use mean fields productively for control of a large number of unmanned aerial
vehicles.

[@gregor]:
  https://www.semanticscholar.org/paper/Using-M-Embeddings-to-Learn-Control-Strategies-for-Gebhardt-H%C3%BCttenrauch/9f550815f8858e7c4c8aef23665fa5817884f1b3
[@meanfielduav]: https://ieeexplore.ieee.org/abstract/document/9137257

#### Aggregation with other pooling functions

Instead of using the empirical mean, other aggregation methods can also be used:

Max pooling:

$$ψ_O = \max_{o_i ∈ O} o_i$$

Softmax pooling:

$$ψ_O = \sum_{o_i ∈ O} o_i \frac{e^{o_i}}{\sum_{o_j ∈ O} e^{o_j}}$$

Max-pooling is widely used in convolutional neural networks to reduce the image
dimensionality where it consistently outperforms mean (average) pooling. Softmax
aggregation was used by @{https://arxiv.org/abs/1703.04908} for MARL.

#### Bayesian Gaussian conditioning

Aggregation with Gaussian conditioning works by starting from a Gaussian prior
distribution and updating it using a probabilistic observation model for every
seen observation. Gaussian conditioning is widely used in applications such as
Gaussian process regression
[@{https://link.springer.com/chapter/10.1007/978-3-540-28650-9_4}]. Bayesian
aggregation was used by @bayesiancontextaggregation for context aggregation in
conditional latent variable (CLV) models. There is no related work using
Bayesian aggregation for MARL.

#### Attention mechanisms

Attention mechanisms are commonly used in other areas of machine learning. The
multi-head attention mechanism was introduced by
@{https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html}
and is the core of the state-of-the-art model for many natural language
processing tasks.

For attentive aggregation, the observations from the different agents are first
attented to by some attention mechanism, then the resulting new feature vector
is aggregated using some aggregation mechanism.

<!-- {attention: 30,31,32,33 (from ARE paper)} -->

We only consider the specific multi-head residual masked self attention variant
of attention mechanisms used by @openai. The general multi-head attention
($\text{MHA}$) mechanism was introduced by
@{https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html}.
It uses multiple independent attention heads to compute weights for a query, a
key, and a value.

For self-attention, the query $Q$, the key $K$ and the value $V$ are all set to
the input value. _Residual_ means that the input is added to the output of the
attention mechanism, so the attention mechanism only has to learn a _residual_
value that modifies the input features. In [@openai], the authors combine the
multi-head attention with a mean aggregation. Note that they only use the
attention mechanism to individually transform the information from each separate
observable into a new feature set instead of using it as a weighing function for
the aggregation.

@{http://proceedings.mlr.press/v97/iqbal19a.html} use a different approach with
a decentralized policy and a centralized critic. An attention mechanism is used
to weigh the features from the other agents, then aggregates them to calculate
the value function.

@are first encode all aggregatable observations together with the proprioceptive
observations with an "extrinsic encoder", then compute attention weights using
another encoder from that and aggregate the features retrieved from a separate
"intrinsic encoder" using those weights.

[@are]: https://ieeexplore.ieee.org/document/9049415

#### Other aggregation methods

@maxpaper also did experiments with aggregation into histograms and aggregation
with radial basis functions, though the results indicated they were outperformed
by a neural network encoder with mean aggregation.

<!-- @tocommunicate and
- based on temporal information (recurrent nns)
-

-->
