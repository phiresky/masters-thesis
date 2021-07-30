Multi-agent reinforcement learning (MARL) is an emerging field in reinforcement
learning with real world applications such as unmanned aerial vehicles,
search-and-rescue, and warehouse organization. There are many different
approaches for applying the methods used for single-agent reinforcement learning
to MARL. In this work, we survey different learning methods and environment
properties and then focus on a problem that persists through most variants of
MARL: How should one agent use the information gathered from a large and varying
number of observations of the world in order to make decisions? We focus on
three different methods for aggregating observations and compare them regarding
their training performance and sample efficiency. We introduce a policy
architecture for aggregation based on Bayesian conditioning and compare it to
mean aggregation and attentive aggregation used in related work. We show the
performance of the different methods on a set of cooperative tasks that can
scale to a large number of agents, including tasks that have other objects in
the world that need to be observed by the agents in order to solve the task.

We optimize the hyperparameters to be able to show which parameters lead to the
best results for each of the methods. In addition, we compare different variants
of Bayesian aggregation and compare the recently introduced Trust Region Layers
learning method to the commonly used Proximal Policy Optimization.
