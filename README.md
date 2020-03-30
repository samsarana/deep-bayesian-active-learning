# Deep Bayesian Active Learning
Reproducibility assessment of the machine learning paper [Deep Bayesian Active Learning with Image Data](https://arxiv.org/pdf/1703.02910.pdf).

I'm currently running the experiments ([`exp1.sh`](https://github.com/samsarana/deep-bayesian-active-learning/blob/master/exp1.sh)) and will post the results here when they've finished.

### Summary
The key contribution of the paper is to adapt four popular active learning (AL) methods for use with deep neural networks. Prior to this paper, this had been challenging because AL methods typically rely on being able to represent model uncertainty, yet deep NNs do not (by default) provide this information. The authors solve this problem by applying the [*MC-Dropout*](https://arxiv.org/pdf/1506.02142.pdf) algorithm, which provides a practical way to extract model uncertainty from deep NNs.

To provide empirical evidence that their solution works, they test it on two tasks: [MNIST image classification](http://yann.lecun.com/exdb/mnist/) and [skin cancer diagnosis from lesion images](https://challenge.kitware.com/#challenge/560d7856cad3a57cfde481ba). In the MNIST experiment, they find that three of the four methods outperform acquiring examples at random. I'll update this with my results when the experiments are finished (they're quite compute-intensive, taking ~8 days on my VM with 8 vCPUs).

### But what's active learning, anyway?
Active learning is a framework for training a machine learning model to a particular accuracy while minimising the need for labelled data. You can read more about it in section 1 of [this paper](https://arxiv.org/pdf/1703.02910.pdf), or check out my [intuitive introduction to active learning](https://samsarana.github.io/posts/2020/03/27/active-learning.html).
