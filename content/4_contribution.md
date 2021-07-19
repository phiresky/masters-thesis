## Our Contribution {#sec:contribution}

We introduce a policy architecture for deep reinforcement learning that projects
observations from one or more different kinds of observables into samples of
latent spaces, then aggregates them into a single latent value. This makes the
architecture scale to any number as well as a varying number of observables.

### Policy Architecture

![Overview of our general model architecture.](build/model.drawio.pdf){#fig:model}

[@Fig:model] shows a schematic of the general model architecture we describe
below.

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
observations $p$ to get a single encoded value $e_k$.

$$e_k = (p, G_1, G_2, ...)$$

This value is then passed through a decoder that consists of one or more dense
layers. Finally the decoded value is transformed to the dimensionality of the
action space or to $1$ to get the output of the value function. While we share
the architecture for the policy and value function, we use a separate copy of
the compute graph for the value function, so the weights and training are
completely independent.

The tasks we consider have a continuous action space. We use a diagonalized
gaussian distribution where the mean $μ$ of each action is output by the neural
network while the variance of each action is a free-standing learnable variable
only passed through $\exp$ or $\text{softplus}$ to ensure positivity.

#### Mean/Max/Softmax Aggregation

Each sample in the latent space is weighted by a function $\text{weigh}()$ and
aggregated using an aggregation operator $\bigoplus$.

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

For the bayesian aggregation, we introduze $z$ as the aggregated latent variable
with $$e_{k→G}=:μ_z.$$ $z$ is seen as a random variable with a Gaussian
distribution:

$$z \sim \mathcal{N}(μ_z,σ_z^2)\quad (p(z) ≡ \mathcal{N}(μ_z,σ_z^2))$$

This random variable is estimated using a diagonal gaussian prior as an a-priori
estimate:

$$p_0(z)≡\mathcal{N}(μ_{z_0}, diag(σ_{z_0}^2))$$

This prior is then updated with Bayesian conditioning using each of the observed
elements $o_{k→g_i}$ in the aggregation group. We interpret each observation as
a new sample from the distribution $p(z)$, each with a mean $r_{k→g_i}$ and a
standard deviation $σ_{r_{k→g_i}}$. We use the probabilistic observation model
and consider the conditional probability
$$p(r_{k→g_i}|z) ≡ \mathcal{N}(r_{k→g}, σ_{r_{k→g}}^2)$$.

With Bayes' rule, we can invert this conditional probability to get:
$$p(z|r_{k→g_i}) = \frac{p(r_{k→g_i}|z) p(z)}{p(r_{k→g_i})}$$

$$p(z) = \frac{p(z|r_{k→g_i}) p(r_{k→g_i})}{p(r_{k→g_i}|z)}$$

When considering the prior together with each observation in the observation
group as an update we get the following closed form description of $z$:

$$\sigma_z^2 = \frac{1}{\frac{1}{\sigma_{z_0}^2} + \sum_{i=1}^n{\frac{1}{\sigma_{r_i}^2}}}$$

$$\mu_z = \mu_{z_0} + \sigma_z^2 \cdot \sum_{i=1}^{n}{\frac{(r_n-\mu_{z_0})}{\sigma_{r_i}^2}}$$

This derivation is based on [@bayesiancontextaggregation, sec. 7.1], which is
based on [@{https://www.springer.com/gp/book/9780387310732}, sec. 2.3.3].

The values of $μ_{z_0}$ and $σ_{z_0}^2$ are learned as free-standing variables
using the backpropagation during training:

$$r_{k→g_i} = \text{enc}_r(o_{k→g_i}), \quad σ_{r_{k→g_i}} = \text{enc}_σ(o_{k→g_i})$$

$enc_r$ and $enc_σ$ are either separate dense neural networks or a single dense
neural network with the output having two outputs per feature (one for the mean,
one for the variance).

A graphical overview of this method is shown in [@fig:bayesianagg].

![Bayesian Aggregation in graphical form. The output is concatenated with the other observations and passed to the decoder.](build/bayesianagg.drawio.pdf){#fig:bayesianagg}

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

![A schematic of the model architecture used by OpenAI [@openai] using masked residual self-attention. It is similar to our architecture ([@fig:model]) except for the LIDAR in the self-observations as well as the LSTM layer at the end.](build/model-openai.drawio.pdf){#fig:openai}

### Multi-agent learning with PPO

PPO and other policy gradient methods are designed for single-agent
environments. We adapt PPO to the multi-agent setting without making any major
changes: The policy parameters are shared for all agents, and each agent gets
the same global reward. The value function for advantage estimation is based on
the same architecture as the policy. Since the stable-baselines3 implementation
of PPO is already written for vectorized environments (collecting trajectories
from many environments running in parallel), we create a new VecEnv
implementation that flattens multiple agents in multiple environments.

During training, the data of each agent is collected as if that agent was the
only agent in the world. For example, a batch of a single step of 10 agents each
in 20 different environment becomes 200 separate training samples. Each agent
still only has access to its own local observations, not the global system
state. This means that during inference time, each agent has to act
independently, based on the observations it makes locally.
