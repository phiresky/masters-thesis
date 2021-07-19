## Preliminaries and related work {#sec:preliminaries}

### Policy Gradient training methods

#### Proximal Policy Optimization (PPO)

Proximal Policy Optimization [@{https://arxiv.org/abs/1707.06347}] is a commonly
used policy gradient method for reinforcement learning. In PPO, each training
step consists of collecting a set of trajectory rollouts, then optimizing a
"surrogate" objective function using stochastic gradient descent. The surrogate
objective function approximates the policy gradient while enforcing a trust
region by clipping the update steps. PPO is a successor to Trust Region Policy
Optimization (TRPO) with a simpler implementation and empirically better
performance.

Since the performance of PPO depends on a number of implementation details and
quirks, we use the stable-baselines3 [@stable-baselines3] implementation of
PPO-Clip, which has been shown to perform as well as the original OpenAI
implementation
[@{https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#results}].

We use PPO as the default for most of our experiments since it is widely used in
other deep reinforcement learning work.

#### Trust Region Layers (PG-TRL) {#sec:trl}

Differentiable trust region layers are an alternative method to enforce a trust
region during policy updates introduced by [@trl]. PPO uses a fixed clipping
ratio to enforce the trust region, which can result in unstable training.
Instead of using a fixed clipping boundary, in PG-TRL a surrogate policy is
created with a new layer at the end that projects the unclipped policy back into
the trust region. To prevent the real policy and the projected policy from
diverging, the divergence between them is computed in the form of the KL
divergence, the Wasserstein L2 distance, or the Frobenius norm. The computed
divergence is then added to the policy gradient loss function with a high
factor. {todo: more detail?}

We explore PG-TRL as an alternative training method to PPO.

### Reinforcement Learning in multi-agent systems (MARL)

There are many variants of applying reinforcement learning to multi-agent
systems.

An overview over recent MARL work and some of the differing properties can be
found in [@{https://arxiv.org/abs/1911.10635}] and
[@{https://www.mdpi.com/2076-3417/11/11/4948}].

In our experiments, we impose a set of restrictions on the environments and
learning process. The restrictions we impose here are mostly based on
[@maxpaper]. Below, we describe the major differing factors of both the learning
process and the environments, as well as the variants we choose to consider.

<!-- [@tocommunicate]:
  https://proceedings.neurips.cc/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html

- Learning to Communicate with Deep Multi-Agent Reinforcement Learning
  [@tocommunicate] -->

<!-- - Multi-agent Reinforcement Learning as a Rehearsal for Decentralized Planning
  https://www.sciencedirect.com/science/article/abs/pii/S0925231216000783 -->

##### Homogenous vs heterogenous agents

In general, the agents in a multi-agent environments can differ in their
intrinsic properties. For example, they can have different control dynamics,
maximum speeds, different observation systems, or different possible actions. We
only consider environments with homogenous agents: All agents have the same
physical properties, observation space, and action space. They only differ in
their extrinsic properties: Their current position, rotation, and speed. This
also causes them to have a different perspective, different observations and
thus different actions, resulting in differing behavior even when they are
acting according to the same policy.

##### Centralized learning

During training, we can either learn the agent policies in a centralized or a
decentralized fasion.

In decentralized learning each agent learns their own policy, while in
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
@{https://link.springer.com/chapter/10.1007/978-3-319-71682-4_5}]. We only
consider the CLDE case.

##### Decentralized execution

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

##### Cooperative, adversarial, team-based

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
adversarial task is Go [@{https://www.nature.com/articles/nature24270}; ] as
well as many other board games, though these usually have a fairly small number
of agents.

Team-based tasks have multiple teams that each have a conflicting goal, but from
the perspective of each team the task is cooperative (with a single reward
function). The team reward can either be defined directly for the team or by
averaging a reward function of each team member.

<!-- One example of a team-based environment is [@{https://openai.com/blog/openai-five/}]. -->

An example of a team-based environment is the OpenAI Hide-and-Seek Task
[@openai]. In this task there are two teams of agents in a simulated physical
world with obstacles, and members of one team try to find the members of the
other team. The Hide-team is rewarded +1 if none of the team members is seen by
any seeker, and -1 otherwise. The Seek-team is given the opposite reward.

In our case, we only consider cooperative environments, and we use the same
reward function for all agents. Real-world multi-agent tasks are usually
cooperative since in adversarial environments, one entity would not have control
over multiple adversarial parties.

##### Local observations, partial visibility

The observations that each agent receives in our experiments are local. For
example, if one agent sees another, that agent's properties are observed
relative to the current agent - the distance, relative bearing, and relative
speed.

In addition, each agent may only have local visibility, for example it can only
observe the positions of agents and objects in the world within some radius or
hindered by obstacles. In this work we focus on global visibility since the
additional noise introduced by local observability would be detrimental to the
quality of our results.

<!-- We consider both global visibility as well as local visibility cases. -->

##### Simultaneous vs turn-based

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

#### MARL tasks in related work

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

### Aggregation methods

The observations of each agent in a MARL task contains a varying number of
observables. The observables can be clustered into groups where each observable
is of the same kind and shape. For example, one observable group would contain
all the neighboring agents, while another would contain e.g. a specific type of
other observed objects in the world.

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
the encoded samples to retrieve a representation of the system state $ψ$ based
on all observed observables.

$$ψ_O = μ_O = \frac{1}{|O|} \sum_{o_i ∈ O} \text{encode}(o_i)$$

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
