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

$$e_{k→g} = \frac{1}{n} \sum_{i=1}^n \text{resatt}(\text{enc}(o_i))$$

with $\text{enc}$ being a dense neural network with zero or more hidden layers,
and

$$\text{resatt}(o_i) = o_i + \text{dense}(\text{MHA}(o_i, o_i, o_i)).$$

_Residual_ means that the input is added to the output of the
attention mechanism, so the attention mechanism only has to learn a _residual_
value that modifies the input features. 

The $\text{MHA}$ module is the multi-head attention module from
[@{https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html},
sec. 3.2.2] as described in @sec:mha.

This method of attentive aggregation is similar to the method successfully used
by @openai. An overview over the model architecture used in [@openai] can be
seen in [@fig:openai].

![A schematic of the model architecture used by OpenAI [@openai] using masked residual self-attention. It is similar to our architecture ([@fig:model]) except for the LIDAR in the self-observations as well as the LSTM layer at the end.](images/model-openai.drawio.svg){#fig:openai}