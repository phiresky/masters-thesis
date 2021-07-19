## Introduction

Ant colonies, bee swarms, fish colonies, migrating birds all exhibit swarming
behaviour to achieve a common goal or to gain an advantage over what's possible
alone. Each individual within a swarm usually only has limited perception, and
often there is no central control that enforces a common goal or gives
instructions. Swarms of animals can self-organize and exhibt complex emergent
behaviour in spite of the limited intelligence, limited perception, and limited
strength of every individual.

In recent years and with the advent of deep reinforcement learning, it has
become possible to create artificial agents that solve fairly complex tasks both
in simulated environments as well as the real world, even when the path to the
goal is difficult to identify and the reward is sparse. Most of the research
however is focused on a single agent interacting with a world, and giving the
single agent as much power and flexibility as it needs to solve the task. Akin
to animal swarms, which often consist of fairly simple and limited individuals,
we focus on simple artificial agents that need to cooperate to solve a common
task, where the task is either difficult or impossible for a single one of the
agents. The agents thus need to be able to learn to work together and to act
even with the limited information they are able to perceive.

Swarms of artificial agents can have multiple applications, some inspired by
swarms in the animal kingdom, some going beyond. Robots can be used in logistics
to organize warehouses. Swarms of robots can be used to map out dangerous areas,
solve mazes, find trapped or injured people in search-and-rescue missions
quicker and more safely than humans. Swarms of walking robots could be used for
animal shepherding [@{https://ieeexplore.ieee.org/abstract/document/9173524}].
Drone swarms can be used for entertainment in light shows, for advertising, for
autonomous surveillance, or they could be weaponized and used as military
robots. Further in the future, swarms of nanobots could be used in medicine to
find and target specific cells in the body. Swarms of autonomous spacecraft can
be used to explore space, to harvest resources, and to terraform planets into
paperclips.

Swarms of homogenous agents can be very resilient. Since no one individual is a
critical component of the swarm as a whole, individual failures do not result in
a critical collapse of the whole swarm.

Robot swarms can be controlled by explicit algorithmic behaviour, but that can
be complicated, inflexible and fragile. Controlling robot swarms using policies
learned with deep reinforcement learning has promising results in recent
literature such as [@openai].

To apply deep reinforcement learning to multi-agent systems, we need to make a
few adjustments to the existing learning algorithms and figure out how best to
design the policy network. Specifically, we need a way to feed a large and
varying amount of observations from the neighboring agents into the fixed size
input of a dense neural network. In this work, we consider three aggregation
methods on a set of different multi-agent tasks: Mean aggregation, Bayesian
aggregation, and attentive aggregation. Our main goal is to compare these
methods with regards to their training performance and sample efficiency.

We first give an overview over related literature in @sec:relatedwork and an
overview of all the preliminaries we need for our work in @sec:preliminaries,
including the environments / tasks we consider, the background for the
aggregation methods we use, and the reinforcement learning algorithms. Then we
describe the details of our contribution in @sec:contribution with details about
the policy architecture and the different aggregation methods. Our experimental
setup, including the specific environments we use to carry out our experiments
are described in @sec:experiments. Finally we show and interpret the results of
our experiments in @sec:results and talk about the conclusions we can draw from
the experiments, as well as the potential for future work in @sec:conclusion.
