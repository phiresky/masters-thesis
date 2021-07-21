---
title: Bayesian and Attentive Aggregation for Cooperative Multi-Agent Deep Reinforcement Learning
keywords:
- MARL
- multi-agent
- mreinforcement learning
- Bayesian conditioning
lang: en-US
date-meta: '2021-07-21'
author-meta:
- Robin Ruede
cref: true
figPrefix:
- Figure
- Figures
eqnPrefix:
- Equation
- Equations
tblPrefix:
- Table
- Tables
secPrefix:
- Section
- Sections
linkReferences: true
nameInLink: true
header-includes: |-
  <!--
  Manubot generated metadata rendered from header-includes-template.html.
  Suggest improvements at https://github.com/manubot/manubot/blob/main/manubot/process/header-includes-template.html
  -->
  <meta name="dc.format" content="text/html" />
  <meta name="dc.title" content="Bayesian and Attentive Aggregation for Cooperative Multi-Agent Deep Reinforcement Learning" />
  <meta name="citation_title" content="Bayesian and Attentive Aggregation for Cooperative Multi-Agent Deep Reinforcement Learning" />
  <meta property="og:title" content="Bayesian and Attentive Aggregation for Cooperative Multi-Agent Deep Reinforcement Learning" />
  <meta property="twitter:title" content="Bayesian and Attentive Aggregation for Cooperative Multi-Agent Deep Reinforcement Learning" />
  <meta name="dc.date" content="2021-07-21" />
  <meta name="citation_publication_date" content="2021-07-21" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Robin Ruede" />
  <meta name="citation_author_institution" content="Autonomous Learning Robots Lab, Karlsruhe Institute of Technology" />
  <link rel="canonical" href="https://phiresky.github.io/masters-thesis/" />
  <meta property="og:url" content="https://phiresky.github.io/masters-thesis/" />
  <meta property="twitter:url" content="https://phiresky.github.io/masters-thesis/" />
  <meta name="citation_fulltext_html_url" content="https://phiresky.github.io/masters-thesis/" />
  <meta name="citation_pdf_url" content="https://phiresky.github.io/masters-thesis/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://phiresky.github.io/masters-thesis/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://phiresky.github.io/masters-thesis/v/c6fd0f74c391ec3963776dd6248da7d6e7e4808d/" />
  <meta name="manubot_html_url_versioned" content="https://phiresky.github.io/masters-thesis/v/c6fd0f74c391ec3963776dd6248da7d6e7e4808d/" />
  <meta name="manubot_pdf_url_versioned" content="https://phiresky.github.io/masters-thesis/v/c6fd0f74c391ec3963776dd6248da7d6e7e4808d/manuscript.pdf" />
  <meta property="og:type" content="article" />
  <meta property="twitter:card" content="summary_large_image" />
  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />
  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />
  <meta name="theme-color" content="#ad1457" />
  <!-- end Manubot generated metadata -->
bibliography:
- content/manual-references.json
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: ci/cache/requests-cache
manubot-clear-requests-cache: false
...






<small><em>
This manuscript
([permalink](https://phiresky.github.io/masters-thesis/v/c6fd0f74c391ec3963776dd6248da7d6e7e4808d/))
was automatically generated
from [phiresky/masters-thesis@c6fd0f7](https://github.com/phiresky/masters-thesis/tree/c6fd0f74c391ec3963776dd6248da7d6e7e4808d)
on July 21, 2021.
</em></small>

## Authors



+ **Robin Ruede**<br>
    · ![GitHub icon](images/github.svg){.inline_icon}
    [phiresky](https://github.com/phiresky)<br>
  <small>
     Autonomous Learning Robots Lab, Karlsruhe Institute of Technology
  </small>


## Abstract {.page_break_before}


Multi-agent reinforcement learning (MARL) is an emerging field in reinforcement
learning with real world applications such as unmanned aerial vehicles,
search-and-rescue, and warehouse organization. There are many different
approaches for applying the methods used for single-agent reinforcement learning
to MARL. In this work, we survey different learning methods and environment
properties and then focus on a problem that persists through most variants: How
should one agent use the information gathered from a large and varying number of
observations of the world in order to make decisions? We focus on three
different methods for aggregating observations and compare them regarding their
training performance and sample efficiency. We introduce a policy architecture
for aggregation based on Bayesian conditioning and compare it to mean
aggregation and attentive aggregation used in related work. We show the
performance of the different methods on a set of cooperative tasks that can
scale to a large number of agents, including tasks that have other objects in
the world that need to be observed by the agents in order to solve the task.

We optimize the hyperparameters to be able to show which parameters lead to the
best results for each of the methods. In addition, we compare different variants
of Bayesian aggregation and compare the recently introduced Trust Region Layers
learning method to the commonly used Proximal Policy Optimization.


## Introduction

Ant colonies, bee swarms, fish colonies, and migrating birds all exhibit
swarming behaviour to achieve a common goal or to gain an advantage over what's
possible alone. Each individual within a swarm usually only has limited
perception, and often there is no central control that enforces a common goal or
gives instructions. Swarms of animals can self-organize and exhibt complex
emergent behaviour in spite of the limited intelligence, limited perception, and
limited strength of every individual.

In recent years and with the advent of deep reinforcement learning, it has
become possible to create artificial agents that solve fairly complex tasks both
in simulated environments as well as the real world, even when the path to the
goal is difficult to identify and the reward is sparse. Most of the research,
however, is focused on a single agent interacting with a world, and giving the
single agent as much power and flexibility as it needs to solve the task. Akin
to animal swarms, which often consist of fairly simple and limited individuals,
we focus on simple artificial agents that need to cooperate to solve a common
task, where the task is either difficult or impossible for an individual agent.
The agents thus need to be able to learn to work together and to act even with
the limited information they are able to perceive.

Swarms of artificial agents can have multiple applications, some inspired by
swarms in the animal kingdom, some going beyond. Robots can be used in logistics
to organize warehouses. Swarms of robots can be used to map out dangerous areas,
solve mazes, find trapped or injured people in search-and-rescue missions
quicker and more safely than humans
[@{https://link.springer.com/article/10.1007/s43154-021-00048-3}]. Swarms of
walking robots could be used for animal shepherding
[@{https://ieeexplore.ieee.org/abstract/document/9173524}]. Drone swarms can be
used for entertainment in light shows
[@{https://doi.org/10.3929/ethz-a-010831954}], for advertising, for autonomous
surveillance [@{https://doi.org/10.1117/12.830408}], or they could be weaponized
and used as military robots [@{https://apps.dtic.mil/sti/citations/AD1039921}].
Further in the future, swarms of nanobots could be used in medicine to find and
target specific cells in the body
[@{https://ieeexplore.ieee.org/document/7568316}]. Swarms of autonomous
spacecraft can be used to explore space, to harvest resources, and to terraform
planets into paperclips
[@{https://link.springer.com/article/10.1007/s00146-018-0845-5}].

Swarms of homogenous agents can be very resilient. Since they can be designed
such that no one individual is a critical component of the swarm as a whole,
individual failures do not necessarily result in a critical collapse of the
whole swarm.

Robots can be controlled by explicit algorithmic behaviour, but that can be
complicated, inflexible and fragile, as is shown by the success in recent
applications of reinforcmeent learning to control tasks that have not been
solved by pre-programmed algorithms [@{https://arxiv.org/abs/1808.00177}].
Controlling robot swarms using policies learned with deep reinforcement learning
has promising results in recent literature such as [@openai].

To apply deep reinforcement learning to multi-agent systems , we need to make a
few adjustments to the existing learning algorithms and figure out how best to
design the policy network. Specifically, we need a way to feed a large and
varying amount of observations from the neighboring agents into the fixed size
input of a dense neural network. In this work, we consider three aggregation
methods on a set of different multi-agent tasks: Mean aggregation, Bayesian
aggregation, and attentive aggregation. Our main goal is to compare these
methods with regards to their training performance and sample efficiency. We
limit ourselves to a specific subset of MARL tasks that are fully cooperative
with a team reward, with homogenous agents in a
centralized-learning/decentralized-execution setup.

We first give an overview over all the preliminaries we need for our work in
@sec:preliminaries, including reinforcement learning in general, multi-agent
reinforcement learning, and the background for the aggregation methods we use.
Next we describe related work in @sec:relatedwork. Then, we describe the details
of our contribution in @sec:contribution with details about the policy
architecture and the different aggregation methods. Our experimental setup,
including the specific environments we use to carry out our experiments are
described in @sec:experiments. Finally, we show and interpret the results of our
experiments in @sec:results and talk about the conclusions we can draw from the
experiments as well as the potential for future work in @sec:conclusion.


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
- The reward function that specifies the immediate reward an agent receives
  depending on the previous state, the action, and the new state

The MDP can either run for an infinite period or finish after a number of
transitions.

The method by which an agent chooses its actions based on the current state is
called a _policy_. The policy defines a probability for taking each action based on a specific state.

Based on a policy we can also define the _value function_, which is the sum of
all future rewards that an agent gets based on an initial state and a specific
policy. The value function can also include an exponential decay for weighing
future rewards.

Often we need to extend MDPs to allow for an observation model: A Partially
observable Markov decision process (POMDP) is an extension to MDPs that
introduces an indirection between the set of states and the input to the
policy (called an _observation_). In the definition of a POMDP a set of
observations is added with a probability of each state leading to a specific
observation. The policy now depends on the observation and the
action instead of the state and the action.

To successfully solve a reinforcement learning task, we need to find a policy
that has a high expected reward - we want to find the _optimal policy function_
that has the highest value function on the initial states of our environment.
Since finding the optimal policy directly is impossible for even slightly
complicated tasks, we instead use optimization techniques.

### Actor-critic training methods

Policy gradient training methods are a reinforcement learning technique that
optimize the parameters of a policy using gradient descent. Actor-critic methods
optimize both the policy and an estimation of the value function at the same time.

There are multiple commonly used actor critic training methods such as Trust
Region Policy Optimization (TRPO)
[@{http://proceedings.mlr.press/v37/schulman15.html}], Proximal Policy
Optimization (PPO) [@ppo] and Soft Actor Critic
[@{https://arxiv.org/abs/1801.01290}]. We run most of our experiments with one of
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

PPO optimizes the policy using

$$θ_{k+1} = \text{argmax}_{θ} E_{s,a \sim π_{θ_k}} [L(s, a, θ_k, θ)]$$

Where $π_{θ_k}$ is a policy with parameters $θ$ in training step $k$, $s$ is the state, $a\sim π_{θ_k}$ is the action distribution according to the the policy at step $k$. L is given by 

$$L(s,a,θ_k,θ) = \min \left( \frac{π_θ(a|s)}{π_{θ_k}(a|s)} A^{π_{θ_k}}(s,a), \text{clip}(\frac{π_θ(a|s)}{π_{θ_k}(a|s)}, 1 - ε, 1 + ε) A^{π_{θ_k}}(s,a) \right).$$

$A$ is the advantage of taking a specific action in a specific state as opposed to the other actions as weighted by the current policy, estimated using Generalized Advantage Estimation [@{https://arxiv.org/abs/1506.02438}] based on the estimated value function.


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
policy. To prevent an infinite stacking of old policies, the explicitly
projected policy is only used as a surrogate, while the real policy as based on
a learned approximation. To prevent the real policy and the projected policy
from diverging, the divergence between them is computed again in the form of the
KL divergence or the Wasserstein $W_2$. The computed divergence (_trust region
regression loss_) is then added to the policy gradient loss function with a high
factor.

We explore PG-TRL as an alternative training method to PPO.

### Multi-agent reinforcement learning (MARL)

DecPOMDP, SwarMDP definition and how to solve it using PPO

### Environment model and learning process

In our experiments, we impose a set of restrictions on the environments and
learning process. The restrictions we impose here are mostly based on
[@maxpaper]. here, we describe the major differing factors of both the learning
process and the environments, as well as the variants we choose to consider.

In general, the agents in a multi-agent environments can differ in their
intrinsic properties. For example, they can have different control dynamics,
maximum speeds, different observation systems, or different possible actions. We
only consider environments with homogenous agents: All agents have the same
physical properties, observation space, and action space. They only differ in
their extrinsic properties, e.g., their current position, rotation, and speed.
This also causes them to have a different perspective, different observations
and thus different actions, resulting in differing behavior even when they are
acting according to the same policy.

We only consider cooperative environments, and we use the same reward function
for all agents. Real-world multi-agent tasks are usually cooperative since in
adversarial environments, one entity would not have control over multiple
adversarial parties.

We focus on global visibility since the additional noise introduced by local
observability would be detrimental to the quality of our results.

For training we use the centralized-learning/decentralized-execution (CLDE)
approach - a shared common policy is learned for all agents, but the policy is
executed by each agent separately.

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
current system state. We use the empirical mean of the samples to retrieve a
representation of the system state $ψ_O$ based on all observed observables, as
shown in @eq:meanagg.

$$ψ_O = μ_O = \frac{1}{|O|} \sum_{o_i ∈ O} o_i$$ {#eq:meanagg}

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

#### Bayesian aggregation {#sec:bayesianagg1}

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


## Related work {#sec:relatedwork}

There are many variants of applying reinforcement learning to multi-agent
systems.

An overview over recent MARL work and some of the differing properties can be
found in [@{https://arxiv.org/abs/1911.10635}] and
[@{https://www.mdpi.com/2076-3417/11/11/4948}].

<!-- [@tocommunicate]:
  https://proceedings.neurips.cc/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html

- Learning to Communicate with Deep Multi-Agent Reinforcement Learning
  [@tocommunicate] -->

<!-- - Multi-agent Reinforcement Learning as a Rehearsal for Decentralized Planning
  https://www.sciencedirect.com/science/article/abs/pii/S0925231216000783 -->

#### Centralized vs decentralized learning

During training, the agent policies car either be learned in a centralized or a
decentralized fashion.

In decentralized learning, each agent learns their own policy, while in
centralized learning, the policy is shared between all agents. Decentralized
learning has the advantage that the agents do not need to be homogenous, since
the learned policy can be completely different. It also means that it's harder
for the agents to learn to collaborate though, since they are not intrinsically
similar to each other. Decentralized learning is used in
[@{https://arxiv.org/abs/1806.00877};
@{https://ieeexplore.ieee.org/document/6415291};
@{http://proceedings.mlr.press/v80/zhang18n.html}].

Centralized learning means the training process happens centralized, with a
single policy. It has the advantage of only needing to train one policy network
and the possibility of being more sample-efficient. Centralized learning
requires homogenous agents since the policy network parameters are shared across
all agents. It is used for example in [@{https://arxiv.org/abs/1705.08926};
@{https://link.springer.com/chapter/10.1007/978-3-319-71682-4_5}].

#### Centralized vs decentralized execution

When using centralized learning, it is possible to use a single policy to output
actions for all agents at the same time. This is called "centralized execution".
The policy chooses the actions based on the global system state in a "birds-eye"
view of the world. The disadvantage is that this means agents can't act
independently if the environment has restrictions such as partial observability,
and it is less robust to communication failures and to agent "deaths".

The alternative is decentralized execution - the observation inputs and action
outputs of the policy networks are local to a single agent. This can formally be
described as a _Decentralized Partially Observable Markov Decision Process_
(Dec-POMDP) [@{https://link.springer.com/chapter/10.1007/978-3-642-27645-3_15}].
When combining centralized learning with decentralized execution, there is only
one policy network that is shared between all agents but the inputs and outputs
of the policy are separate.

Centralized-learning with decentralized-execution is thus a compromise between
performance, robustness, and sample efficiency. CLDE is used e.g. by [@mpe;
@maxpaper; @{https://link.springer.com/chapter/10.1007/978-3-319-71682-4_5};
@{https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17193};
@{https://proceedings.neurips.cc/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html}].

#### Cooperative, adversarial, team-based

Multi-agent environments can be cooperative, adversarial, or team-based.
Cooperative environments are those where all the agents have one common goal. In
adversarial tasks each agent has their own independent goal that conflicts with
the goal of the other agents. An example for a cooperative environment is the
rendezvous task: All agents need to meet up at a single point, where the agents
have to decide independently on the location. The reward here is the negative
average pairwise distance of the agents, see [@sec:rendezvous]. Note that a
cooperative environments can also include other adversarial agents, as long as
those agents are controlled by an explicit algorithm and not a learned policy.
From the perspective of the policy, these agents are considered part of the
environment.

Cooperative learning can be done with separate agent rewards or a single common
reward. Cooperative learning with a single reward means the reward must be
somewhat sparse, since each agent cannot easily judge what the impact of it's
own actions were on the reward in any given time step. This also makes it
impossible for an agent to gain an egoistic advantage over the other agents,
enforcing the learned policy to become Pareto optimal. Another approach was
introduced by @{https://ieeexplore.ieee.org/document/4399095}, who create a
compromise between the sample-efficiency of individual rewards and the
pareto-optimality of a common reward for cooperative MARL settings with their
_Hysteretic Q-Learning_ algorithm that jointly optimizes both an individual as
well as a common reward.

Adversarial environments are usually zero-sum, that is the average reward over
all agents is zero. An example for an adversarial environment is the _Gather_
task described by @{https://ojs.aaai.org/index.php/AAAI/article/view/11371}: In
a world with limited food, agents need to gather food to survive. They can also
kill each other to reduce the scarcity of the food. Another example of an
adversarial task is Go [@{https://www.nature.com/articles/nature24270}] as well
as many other board games, though these usually have a fairly small number of
agents.

Team-based tasks have multiple teams that each have a conflicting goal, but from
the perspective of each team the task is cooperative (with a single reward
function). The team reward can either be defined directly for the team or by
averaging a reward function of each team member.

<!-- One example of a team-based environment is [@{https://openai.com/blog/openai-five/}]. -->

An example of a team-based environment is the OpenAI Hide-and-Seek Task
[@openai]. In this task, there are two teams of agents in a simulated physical
world with obstacles, and members of one team try to find the members of the
other team. The Hide-team is rewarded +1 if none of the team members is seen by
any seeker, and -1 otherwise. The Seek-team is given the opposite reward.

#### Partial visibility

The observations that each agent receives in our experiments are local. For
example, if one agent sees another, that agent's properties are observed
relative to the current agent - the distance, relative bearing, and relative
speed.

In addition, each agent may only have local visibility, for example it can only
observe the positions of agents and objects in the world within some radius or
the visibility can be hindered by obstacles. In [@maxpaper] both the local
and global visibility variants of the same tasks were considered.

#### Simultaneous vs turn-based

In general, multi-agent environments can be turn-based or simultaneous. In
turn-based environments each agent acts in sequence, with the world state
changing after each turn. In simultaneous environments, all agents act "at the
same time", i.e. the environment only changes after all agents have acted in one
time step.

Simultaneous environments can be considered a subset of turn-based environments,
since a simultaneous environment can be converted to a turn-based one by fixing
the environment during turns and only applying the actions of each agent after
the time step is finished. For this reason the API of PettingZoo [@pettingzoo]
is based around turn-based environments. Examples of turn-based environments
include most board games and many video games. Most real world scenarios are
simultaneous though, so we only consider environments with simultaneous actions.

### MARL tasks in related work

@openai create a team-based multi-agent environment with one team of agents
trying to find and "catch" the agents of the other team.

@are test their attention-based architecture on three environments: (1) A
catching game in a discrete 2D world, where multiple paddles moving in one
dimension try to catch multiple balls that fall down the top of the screen. (2)
A spreading game, where N agents try to cover N landmarks. The world is
continous, but the action space is discrete. This game is similar to the
spreading game defined in [@mpe]. (3) StarCraft micromanagement: Two groups of
units in the StarCraft game battle each other, each unit controlled by an agent.

@pettingzoo create an infrastructure for multi-task environments similar to the
OpenAI Gym and add standardized wrappers for Atari multiplayer games, as well as
a reimplementation of the MAgent environments
[@{https://ojs.aaai.org/index.php/AAAI/article/view/11371}], the
Multi-Particle-Environments [@mpe] and the SISL environments
[@{https://link.springer.com/chapter/10.1007/978-3-319-71682-4_5}].

[@mpe]: https://dl.acm.org/doi/10.5555/3295222.3295385

@maxpaper test mean aggregation on two environments: (1) A task where all agents
need to meet up in one point (rendezvous task). (2) A task where the agents try
to catch one or more evaders (pursuit task). We compare our method on both of
these tasks.

@gregor define a set of environments based on Kilobots, simple small round
robots in a virtual environment together with boxes. The Kilobots are tasked
with moving the boxes around or arranging them based on their color. We compare
our method on both the box assembly as well as the box clustering task.


## Scalable information aggregation for deep MARL policies {#sec:contribution}

We introduce a policy architecture for deep reinforcement learning that projects
observations from one or more different kinds of observables into samples of
latent spaces, then aggregates them into a single latent value. This makes the
architecture scale to any number as well as a varying number of observables.

### Policy Architecture

In general, our policy architecture is based on [@maxpaper] and [@openai].

While the policy architecture and weights are shared for all agents, the inputs
are dependent on the current agent $a_k$, thus leading to different outputs for
the action and value function for each agent.

Depending on the task, the inputs differ. In general, we always have the
proprioceptive observations (how the agent $a_k$ sees itself), and the
observations from the $n$ neighboring agents $a_1,…,a_n$. The observations from
each neighbor have the same shape, but differ from the self-observations since
they can include other information like the distance between $a_k$ and $a_i$,
and they may be missing some information that may not be known to $a_k$. In
addition, there can be additional sets of observations, for example for objects
or scripted agents in the environment. The architecture can handle any amount of
observation sets, but the observations in each set must have the same shape.

We call each of the sets of observations "aggregation groups".

First, we collect the observations for each instance $1<i<n$ of each aggregation
group $G$. These observations can be relative to the current agent. Each
observation in the aggregation group $G$ is thus a function of the agent $a_k$
and the instance $g_i$:

$$o_{k→g_i} = \text{observe}(a_k, g_i)$$

After collecting the observations for aggregation group $g$, the observations
are each encoded separately into a latent embedding space:

$$e_{k→g_i} = \text{enc}(o_{k→g_i})$$

The Encoder $\text{enc()}$ is a dense neural network with zero or more hidden
layers.

After being encoded, we interpret each instance of an aggregation group as one
sample of a latent space that represents some form of the information of the
aggregation group that is relevant to the agent. These samples are then
aggregated using one of the aggregation methods described below to get a single
latent space value for each aggregation group:

$$e_{k→G} = \text{agg}_{i=0}^n(e_{k→g_i})$$

We then concatenate all of the latent spaces as well as the proprioceptive
observations $p$ to get a single encoded value $e_k$:

$$e_k = (p, G_1, G_2, ...)$$

This value is then passed through a decoder that consists of one or more dense
layers. Finally, the decoded value is transformed to the dimensionality of the
action space or to $1$ to get the output of the value function. While we share
the architecture for the policy and value function, we use a separate copy of
the compute graph for the value function, so the weights and training are
completely independent.

@Fig:model shows a schematic of the general model architecture described above.

![A schematic of our general model architecture for deep MARL policies with scalable information aggregation. The observation inputs consist of the agent itself and multiple aggregatable observation groups. The observations from the aggregation groups are each passed though an encoder and aggregated with an aggregator. Afterwards all the aggregated observations are concatenated and decoded to get the policy and value function.](images/model.drawio.svg){#fig:model}

The tasks we consider have a continuous action space. We use a diagonalized
Gaussian distribution where the mean $μ$ of each action is output by the neural
network while the variance of each action is a free-standing learnable variable
only passed through $\exp$ or $\text{softplus}$ to ensure positivity.

#### Mean/Max/Softmax Aggregation

Each sample in the latent space is weighted by a function $\text{weigh}()$ and
aggregated using an aggregation operator $\bigoplus$:

$$e_{k→G} = \bigoplus_{i=1}^n \text{weigh}(e_{k→g_i})$$

For mean aggregation, the weighing function multiplies by $\frac{1}{n}$ and the
aggregation operator is the sum:

$$e_{k→G} = \sum_{i=1}^n \frac{1}{n} \cdot (e_{k→g_i})$$

For max aggregation, the weight is $1$ and the aggregation operator takes the
largest value:

$$e_{k→G} = \max_{i=1}^n e_{k→g_i}$$

For softmax aggregation, the weight is based on the softmax function and the
aggregation operator is the sum:

$$e_{k→G} = \sum_{i=1}^n \left(\frac{\exp(e_{k→g_i})}{\sum_{j=1}^n \exp(e_{k→g_j})}\right) e_{k→g_i}$$

#### Bayesian Aggregation {#sec:bayesianagg}

We use a separate latent space and thus a separate observation model $p(z)$ for
each aggregation group.

To make the Bayesian aggregation as described in [@sec:bayesianagg1] work in our
policy network, we need an estimation describing the Gaussian prior ($μ_{z_0}$
and $σ_{z_0}^2$) as well as the observed means $r_i$ and variances $σ_{r_i}^2$.

The mean and variance of the Gaussian prior are learned as free-standing
variables using the backpropagation during training. Both the prior variance as
well as the variance of the observations are rectified to enforce positivity
using $\text{softplus}$.

To get the observed elements $r_i$, we use the two encoder networks $enc_r$ and
$enc_σ$ that consist of a set of dense layers:

$$r_i = \text{enc}_r(o_{k→g_i}), \quad σ_{r_i} = \text{enc}_σ(o_{k→g_i})$$

$\text{enc}_r$ and $\text{enc}_σ$ are either separate dense neural networks or a single dense
neural network with the output having two scalars per feature (one for the mean,
one for the variance). Finally we retrieve the value of $e_{k→G}$ from the
aggregated latent variable, using either just the mean of $z$:

$$e_{k→G} = μ_z$$

or by concatenating the mean and the variance:

$$e_{k→G} = (μ_z, σ_z^2).$$

The mean and variance are calculated from the conditioned Gaussian based on the
encoded observations as described in [@sec:bayesianagg1]:

$$\sigma_z^2 = \frac{1}{\frac{1}{\sigma_{z_0}^2} + \sum_{i=1}^n{\frac{1}{\text{enc}_σ(o_{k→g_i})^2}}}$$

$$\mu_z = \mu_{z_0} + \sigma_z^2 \cdot \sum_{i=1}^{n}{\frac{(\text{enc}_r(o_{k→g_i})-\mu_{z_0})}{\text{enc}_σ(o_{k→g_i})^2}}$$

A graphical overview of this method is shown in [@fig:bayesianagg].

![Bayesian Aggregation in graphical form. The observations from the neighboring agents are encoded with the value encoder and the confidence encoder. They are then used to condition the Gaussian prior estimate of the latent variable $z$ to get the final mean and variance of the a-posteriori estimate. The mean and optionally the variance estimate of $z$ are concatenated with the latent spaces from the other aggregatables and passed to the decoder as shown in @fig:model.](images/bayesianagg.drawio.svg){#fig:bayesianagg}

#### Attentive Aggregation

When using residual self-attentive aggregation, we define the aggregated feature
as

$$e_{k→g} = \frac{1}{n} \sum_{i=1}^n \text{att}(\text{enc}(o_i))$$

with $\text{enc}$ being a dense neural network with zero or more hidden layers,
and

$$\text{att}(o_i) = o_i + \text{dense}(\text{MHA}(o_i, o_i, o_i)).$$

The $\text{MHA}$ module is the multi-head attention module from
[@{https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html},
sec. 3.2.2].

This method of attentive aggregation is similar to the method successfully used
by @openai. An overview over the model architecture used in [@openai] can be
seen in [@fig:openai].

![A schematic of the model architecture used by OpenAI [@openai] using masked residual self-attention. It is similar to our architecture ([@fig:model]) except for the LIDAR in the self-observations as well as the LSTM layer at the end.](images/model-openai.drawio.svg){#fig:openai}

### Multi-agent learning with PPO

PPO and other policy gradient methods are designed for single-agent
environments. We adapt PPO to the multi-agent setting without making any major
changes: The policy parameters are shared for all agents, and each agent gets
the same global reward. The value function for advantage estimation is based on
the same architecture as the policy. Since the stable-baselines3 implementation
of PPO is already written for vectorized environments (collecting trajectories
from many environments running in parallel), we create a new VecEnv
implementation that flattens multiple agents in multiple environments.

Similarily to the setup used for TRPO by @maxpaper, we collect the data of each
agent as if that agent was the only agent in the world. For example, a batch of
a single step of 10 agents each in 20 different environment becomes 200 separate
training samples. Each agent still only has access to its own local
observations, not the global system state. This means that during inference
time, each agent has to act independently, based on the observations it makes
locally.


## Experiments {#sec:experiments}

### Experimental setup

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

### Considered tasks

To evaluate the different aggregation methods we need simulated environments
where multiple agents can cooperatively solve a task. Since most of the commonly
used tasks used to benchmark reinforcement learning algorithms are designed for
a single agent, we use custom built environments.

The following shows which specific tasks we consider.

#### Rendezvous task {#sec:rendezvous}

In the rendezvous task, a set of $n$ agents try to meet up in a single point. An
example is shown in @fig:rendezvous1. Our implementation of the rendezvous task
is modeled after [@maxpaper].

![Visualization of one successful episode of the rendezvous task (from [@maxpaper])](images/rendezvous1.png){#fig:rendezvous1}

The agents are modeled as infinitesimal dots without collisions. They use
double-integrator unicycle dynamics
[@{https://ieeexplore.ieee.org/document/976029}], so the action outputs are the
acceleration of the linear velocity ($\dot{v}$) and the angular velocity
($\dot{ω}$).

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

#### Single-Evader Pursuit task

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
agent and the evader is less than $1%$ of the world width.

##### Multi-evader pursuit

The multi-evader pursuit task is the same as the normal pursuit task, except
there are multiple hard-coded evaders.

The reward here is different, since defining the reward as for the single-evader
task is not obvious. The reward is +1 whenever an evader is caught, 0 otherwise.
An evader is declared caught if the distance to the nearest pursuer is less than
$\frac{2}{100}$ of the world width. Contrary to the single-evader task, the
episode does not end when an evader is caught and instead always runs for 1024
timesteps.

##### Box assembly task

In the box assembly task, the agents are modeled similar to Kilobots, as
described by @gregor. The agents are two-dimensional circles with collision.

For simplicity, we again use single-integrator unicycle dynamics with the linear
velocity $v$ and the angular velocity $ω$ as the action outputs instead of the
specific dynamics of real Kilobots.

The world is a square and contains a few two-dimensional boxes (squares) as
obstacles. For the box assembly task, the goal is to get all the boxes as close
together as possible. Since they agents are much smaller than the boxes, moving
a box is hard for a single agent. The reward is the negative sum of the pairwise
distances between the boxes. We run this task with four boxes and 10 agents.

An example of a successful episode of the task is in @fig:assembly.

![Example successful episode of the box assembly task (from [@gregor])](images/box-assembly.png){#fig:assembly}

[@gregor]: https://www.semanticscholar.org/paper/Using-M-Embeddings-to-Learn-Control-Strategies-for-Gebhardt-H%C3%BCttenrauch/9f550815f8858e7c4c8aef23665fa5817884f1b3

The observation space for the box assembly task contains the following
information:

1. The absolute position (x,y) of the current agent
2. The sin, cos of the absolute rotation of the current agent
3. For every neighboring agent:
   1. The distance between the current agent and the neighbor
   2. The sin and cos of the bearing angle to the neighbor
   3. The sin and cos of the neighbor's rotation relative to our rotation
4. For every neighboring box:
   1. The distance between the current agent and the box
   2. The sin and cos of the bearing angle to the box
   3. The sin and cos of the box's rotation relative to our rotation

##### Box clustering task

The task setup for box clustering is the same as for the box assembly task,
except that each box is assigned a color. The goal is to move the boxes into an
arrangement such that boxes of the same color are as close together as possible,
while boxes of different colors are far away. The reward is the negative sum of
pairwise distances between the boxes of each cluster plus the sum of pairwise
distances of the center of mass of each cluster.

The observation space is the same as the observation space of the box assembly
task, except that each cluster of objects is aggregated into a separate
aggregation space.

{example clustering episode}

<!-- ### Other considered tasks

PettingZoo tasks (without results?)

-->


## Results {#sec:results}

In this section, we present the results of our experiments. We compare (1) the
training performance of the different aggregation methods on various tasks, (2)
aggregating different observation sets into a single vs multiple latent spaces
(3) various variants of Bayesian aggregation (4) PPO with TRL.

In order to compare the performance of different variants, we consider the
average reward per episode over the agent steps used during training. This way
we can evaluate and compare the overall best results as well as the sample
efficiency.

Each policy gradient training step consists of a set of batches of trajectory
rollouts generated using the previous policy. We evaluate the performance after
each training step on a separate set of evaluation environments.

Unless mentioned otherwise, the following setup is used:

- Plotted is the median reward of the n runs at a specific training step, the error band is the 25th and 75th
  percentile of runs
- The activation function used after each layer is LeakyReLU
- For the bayesian aggregation:
  - We only use the mean value $μ_z$ as an output, not $μ_z$ and $σ_z^2$
  - We use a single shared encoder for the value and confidence estimates
  - The a-priori estimate $μ_{z_0}$ is learnable separately for each feature dimension
  - The variance of both the a-priori estimate as well as the encoded estimates are rectified using $\text{softplus}$.
- Multiple aggregatable groups are aggregated into separate latent spaces
- The parameters of the policy and value function are not shared
- The training algorithm used is PPO

### Aggregation method results

We compare the performance of the different aggregation methods (Bayesian
aggregation, mean aggregation, attentive aggregation) on multiple tasks.

We use the notation `64-agg-64` to describe the layer sizes of the neural
network: The numbers before `agg` are the sequential layer sizes of the dense
layers of the encoders of each aggregation group. The numbers after `agg` are
the layer sizes in the decoder after the concatentation of the proprioceptive
observations with the aggregated observations (compare @fig:model).

#### Multi-evader pursuit task

Here, we consider the multi-evader pursuit task with 20 pursuers and 5 evaders on
a torus. @Fig:resmpsmall shows the results of the multi-evader pursuit task with
different aggregation methdos with the same architecture used in [@maxpaper] to
be able to directly compare the results. The architecture is 64-agg-64 with the
tanh activation function. With this architecture, the Bayesian aggregation
performs best.

![Results on the multi-evader pursuit task with the NN architecture adapted from
[@maxpaper]. The Bayesian aggregation performs
best.](images/plots/2021-07-10_13.30.06-Multi-Evader Pursuit
(smallarch).svg){#fig:resmpsmall}

@Fig:resmpopt shows the results with neural network architectures that were separately
hyper-parameter optimized for each aggregation method. The optimized architecture for the mean
aggregation is `174-226-97-agg-96`. The
optimized architecture for the Bayesian aggregation is `120-60-agg-160`. The optimized architecture for the attentive
aggregation is `72-agg-132-200`. All optimized architectures use the LeakyReLU activation functions. Note that the architecture optimized on the
mean aggregation is the deepest with three hidden layers before the aggregation,
while the optimized architecture on the attentive aggregation has multiple
layers after the aggregation instead. With the hyper-parameter optimized
architecture, the mean aggregation performs best. The results are still similar
when using the `120-60-agg-160` architecture for every aggregation method. These
results indicate that the Bayesian aggregation outperforms the mean aggregation
when the neural network is limited in size, but has no advantage when the
neural network is sufficiently large and deep. The neural network seems to be
able to implicitly learn to transform and weigh the information from the
different observables, compensating the advantage of the additional structure of
relevancy / certainty that is given in the Bayesian aggregation.

![Results on the multi-evader pursuit task with the NN architecture hyper-parameter optimized for each aggregation method. The mean aggregation performs best.](images/plots/2021-07-10_16.06.20-Multi-Evader
Pursuit (hpsopt).svg){#fig:resmpopt}

![Like @fig:resmpopt but only the top 1/3 of runs.](images/plots/2021-07-10_16.07.12-Multi-Evader
Pursuit (hpsopt top.33).svg){#fig:resmpopttop}

In addition two the three main aggregation methods we compare in this and the other tasks, we also show a comparison to between mean aggregation and max aggregation on the multi-evader pursuit task in @fig:resaggmax.

![Max aggregation results on the multi-evader pursuit task.](images/plots/2021-07-14_13.43.27-Multi-Evader
Pursuit Max.svg){#fig:resaggmax}

#### Single-evader pursuit task

@Fig:ressp shows the results on the single-evader pursuit task with 10 pursuers
and one evader.

![Results on Single Pursuit task. Architecture: 120-60-agg-160.](images/plots/2021-07-14_13.55.20-Single-evader
Pursuit.svg){#fig:ressp}

#### Rendezvous task

@Fig:resrendezvous.

![Results on the Rendezvous task. Large Architecture: 120-60-agg-160.
Small Architecture: 64-agg-64. Opt Architecture: 146-120-agg-19-177-162](images/plots/2021-07-18_12.34.40-Rendezvous.svg){#fig:resrendezvous}

#### Assembly task

@Fig:resassembly shows the results on the assembly task with ten agents and four
boxes. The three aggregation methods perform very similar, with the attentive
aggregation learning the task slightly quicker.

![Results on the assembly task.](images/plots/2021-07-07_15.07.38-assembly (by
agg method).svg){#fig:resassembly}

#### Clustering task with two clusters

@Fig:resclustering2 shows the results on the clustering task with four boxes
split into two clusters.

![Clustering2 results](images/plots/2021-07-10_18.56.32-Clustering task (2
clusters, by agg method).svg){#fig:resclustering2}

#### Clustering task with three clusters

Doesn't work :(

### Learning algorithm comparison (PPO vs TRL)

In the following, we show some results of the trust region layers training
method (see [@sec:trl]) compared to PPO.

@Fig:resmptrl shows the results on the multi-evader pursuit task. The
architecture are the ones hyper-parameter optimized on PPO on each of the
aggregation methods. TRL seems to show significantly improved training
performance for both the mean and the Bayesian aggregation. @Fig:resmptrltop
shows the same result for only the top one third of runs. The results are very
similar. This indicates that TRL makes the training more stable on this task,
fewer runs fail to achieve the optimal performance.

![TRL vs PPO (multi-evader pursuit)](images/plots/2021-07-11_11.57.49-Multi-Evader
Pursuit TRL.svg){#fig:resmptrl}

![TRL vs PPO (multi-evader pursuit). Top 1/3 of runs](images/plots/2021-07-11_13.23.22-Multi-Evader
Pursuit TRL Top.33.svg){#fig:resmptrltop}

![TRL vs PPO on the assembly task](images/plots/2021-07-11_13.13.37-assembly (by
train algo and agg method).svg)

![TRL vs PPO on single pursuit task](images/plots/2021-07-15_12.57.59-Single-evader
Pursuit TRL vs PPO.svg){#fig:ressptrl}

### Same space vs separate space aggregation

For tasks where we have multiple aggregation groups, we can also aggregate all
observables into the same latent space instead of separate ones. This means that
instead of each aggregation space containing the information of those
aggregatables, the single aggregation space must contain the information of all
observables as it pertains to the current agent. The potential advantage is that
the policy can share more parameters and thus be more sample efficient. It can also scale to a larger number of categories of objects (aggregation groups).
@Fig:sameseparate shows a schematic comparison between the two methods.

In the other experiments we alwas use separate spaces. @Fig:ressameseparate and
@fig:ressameseparate2 show the results of aggregating into a single space vs.
into separate spaces for the multi-evader pursuit and assembly tasks. The
separate-space aggregation performs better.

![Schematic comparison of separate space vs. same-space aggregation. In same-space aggregation the encoders are still separate, but the latent space is shared. The same-space aggregation shares more parameters and can scale better to more aggregation groups.](images/model-sameseparate.drawio.svg){#fig:sameseparate}

![Multi-pursuit same space comparison](images/plots/2021-07-14_13.37.01-Multi-Evader
Pursuit samespace.svg){#fig:ressameseparate}

![Assembly task same space comparison](images/plots/2021-07-14_13.19.13-Assembly
samespace.svg){#fig:ressameseparate2}

### Bayesian aggregation variants

The following shows results for some variants of the Bayesian aggregation.

#### Separate vs common encoder

As described in @sec:bayesianagg, we can either have a shared encoder to predict
the mean and variance of each sample in each aggregation space by making the
last layer of the encoder have two outputs for each feature, or have two fully
separate networks ($enc_r$ and $enc_σ$). In our experiments, using one common
encoder with two outputs generally performs better.

#### Using the aggregated variance or only the mean

In the other experiments with Bayesian aggregation, we only use the predicted
mean of the Gaussian distribution as an input to the decoder:

$$e_{k→G}=μ_z$$

Since the Bayesian aggregation also gives us a full a-posteriori Gaussian
distribution, we also have an estimate of the variance for each feature in the
latent space that is computed from the apriori variance conditioned on each seen
latent space sample. We can feed this variance to the decoder by concatenating
it with the mean predictions in the hope that the neural network is able to use
this additional information:

$$e_{k→G}=(μ_z, σ_z^2)$$

The results of applying this method to the multi-evader pursuit are seen in
@fig:resoutputvariance. The neural network architecture is the same as for the
other experiments with Bayesian aggregation on multi-evader pursuit
(`120-60-agg-160`). Including the variance in the decoder inputs decreases the
performance.

The decreasing performance could be a result of the increased dimension of the decoder inputs. Adding the variance inputs doubles the number of values the decoder has to process and learn from. Since the structure of the encoded values and variances can not known beforehand to the decoder, it has to learn to interpret more information than when receiving just the mean values. The added variance inputs should give the decoder the ability to understand the confidence of each of the value predictions and weigh them accordingly, but the added complexity seems to make it not worth it.

![Results of Bayesian aggregation on the multi-evader pursuit task, depending on whether the variance is also fed into the decoder or only the mean.](images/plots/2021-07-11_12.27.17-Pursuit
(bayes outputvariance).svg){#fig:resoutputvariance}

<!-- ### Local obs aggregation space -->

<!-- ### Activation functions

(probably uninteresting)

-->


<!--# Implementation Details-->


## Conclusion and Future Work {#sec:conclusion}

### Conclusion

We have made a comprehensive comparison between the performance of mean
aggregation, Bayesian aggregation, and attentive aggregation to collect a
varying number of observations on a set of different deep reinforcement learning
tasks. We have observed that there is no clear advantage of one of the methods over
the others, with the results differing strongly between the different tasks.

<!-- In general, the signal to noise ratio of the experiments was pretty low, -->

We have also shown the results of a few variants of the Bayesian aggregation and
concluded that encoding the variance with the same encoder as the estimate,
aggregating into separate latent spaces and not using the aggregated variance as
an input to the decoder achieves the best results.

Finally, we have applied a new training method (trust region layers) to
multi-agent reinforcement learning and compared it to the commonly used PPO. The
results indicate that TRL is usually at least as good as PPO and in some cases
outperforms it.

### Future Work

There are many avenues for future work in this area.

All of our experiments used global visibility due to the noisy nature of limited
local visibility making it harder to make any strong conclusions. Since the
local visibility case also increases the uncertainty of each observation though,
it might be a case where Bayesian aggregation performs better. Future work
should include experiments that have a larger environment with observability
limited by range or by obstacles.

We also only considered tasks with implicit communication - the agents had to
infer the intent of the other agents purely by their actions. There is related
work that adds explicit communication between agents that is learned together
with the policy. This is usually implemented as another action output that is
written directly into the observation of the other agents instead of affecting
the world. Explicit communication architectures may be able to handle some tasks
better than those with implicit communication, but they are often only
applicable to environments with exactly two agents. For more agents, the
performance of explicit communication architectures may be affected by the
aggregation methods used and thus might be a use case for Bayesian aggregation.

- recurrent architecture


\cleardoublepage

## Bibliography {.page_break_before .unnumbered}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>


```url2cite-bibtex
@misc{https://github.com/DLR-RM/stable-baselines3,
  author = {Raffin, Antonin and Hill, Ashley and Ernestus, Maximilian and Gleave, Adam and Kanervisto, Anssi and Dormann, Noah},
  title = {Stable Baselines3},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DLR-RM/stable-baselines3}},
}
```


```url2cite-bibtex
@ARTICLE{https://ieeexplore.ieee.org/document/6415291,  author={Kar, Soummya and Moura, José M. F. and Poor, H. Vincent},  journal={IEEE Transactions on Signal Processing},   title={$Q D$-Learning: A Collaborative Distributed Strategy for Multi-Agent Reinforcement Learning Through  ${\rm Consensus} + {\rm Innovations}$},   year={2013},  volume={61},  number={7},  pages={1848-1862},  doi={10.1109/TSP.2013.2241057}}
```


[@openai]: https://arxiv.org/abs/1909.07528
[@maxpaper]: https://jmlr.org/beta/papers/v20/18-476.html
[@stable-baselines3]: https://github.com/DLR-RM/stable-baselines3
[@bayesiancontextaggregation]: https://openreview.net/forum?id=ufZN2-aehFa
[@trl]: https://openreview.net/forum?id=qYZD-AO1Vn
[@pettingzoo]: https://arxiv.org/abs/2009.14471