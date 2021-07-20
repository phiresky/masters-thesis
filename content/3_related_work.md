## Related work

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

#### Homogenous vs heterogenous agents

In general, the agents in a multi-agent environments can differ in their
intrinsic properties. For example, they can have different control dynamics,
maximum speeds, different observation systems, or different possible actions. We
only consider environments with homogenous agents: All agents have the same
physical properties, observation space, and action space. They only differ in
their extrinsic properties: Their current position, rotation, and speed. This
also causes them to have a different perspective, different observations and
thus different actions, resulting in differing behavior even when they are
acting according to the same policy.

#### Centralized learning

During training, we can either learn the agent policies in a centralized or a
decentralized fasion.

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
@{https://link.springer.com/chapter/10.1007/978-3-319-71682-4_5}]. We only
consider the CLDE case.

#### Decentralized execution

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

In our case, we only consider cooperative environments, and we use the same
reward function for all agents. Real-world multi-agent tasks are usually
cooperative since in adversarial environments, one entity would not have control
over multiple adversarial parties.

#### Local observations, partial visibility

The observations that each agent receives in our experiments are local. For
example, if one agent sees another, that agent's properties are observed
relative to the current agent - the distance, relative bearing, and relative
speed.

In addition, each agent may only have local visibility, for example it can only
observe the positions of agents and objects in the world within some radius or
the visibility can be hindered by obstacles. In this work we focus on global
visibility since the additional noise introduced by local observability would be
detrimental to the quality of our results.

<!-- We consider both global visibility as well as local visibility cases. -->

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
