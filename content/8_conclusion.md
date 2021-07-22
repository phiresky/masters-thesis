# Conclusion and Future Work {#sec:conclusion}

## Conclusion

We have made a comprehensive comparison between the performance of mean
aggregation, Bayesian aggregation, and attentive aggregation to collect a
varying number of observations on a set of different deep reinforcement learning
tasks. We have observed that there is no clear advantage of one of the methods
over the others, with the results differing strongly between the different
tasks.

<!-- In general, the signal to noise ratio of the experiments was pretty low, -->

We have also shown the results of a few variants of the Bayesian aggregation and
concluded that encoding the variance with the same encoder as the estimate,
aggregating into separate latent spaces and not using the aggregated variance as
an input to the decoder achieves the best results.

Finally, we have applied a new training method (trust region layers) to
multi-agent reinforcement learning and compared it to the commonly used PPO. The
results indicate that TRL is usually at least as good as PPO and in some cases
outperforms it.

## Future Work

There are many avenues for future work in this area.

All of our experiments used global visibility due to the noisy nature of limited
local visibility making it harder to make any strong conclusions. Since the
local visibility case also increases the uncertainty of each observation though,
it might be a case where Bayesian aggregation performs better. Future work
should include experiments that have a larger environment with observability
limited by range or by obstacles.

Even though we show in our experiments that Bayesian aggregation into the same
latent space is worse than aggregating into separate spaces, there might be
potential there to be able to scale the number of aggregation groups with more
hyperparameter tuning or by introducing a two-stage encoder where the first
encoder is separate by aggregation group and the second encoder is shared, then
aggregating the output of the second encoder into the same space.

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

<!--
- recurrent architecture -->
