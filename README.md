# Conformalized Quantile Regression with Covariate Shift

Conformal prediction is a framework that can be applied to predictive models to output sets or intervals that achieve a user-chosen error rate under the assumption of exchangeability in the data. We assume familiarity with general techniques of conformal prediction without covariate shift, as in chapters 1 and 2 of [1]. While these techniques are concrete under "nice" conditions, the assumption of exchangeability may fail when the data has undergone a covariate shift.

A covariate shift occurs when the input data $X$ differs between training and testing time, but the conditional distribution of $Y\mid X$, the response when conditioned on the input, remains unchanged. More formally, we assume that the training data comes from a distribution $\mathscr{P}$, the testing data comes from a distribution $\mathscr{P}_{\text{test}}$, and $\mathscr{P}(Y\mid X)=\mathscr{P}_{\text{test}}(Y\mid X)$.

We require $\mathscr{P}_{\text{test}}$ to be absolutely continuous with respect to $\mathscr{P}$, i.e., if a set $A$ is $\mathscr{P}$-null, then it is also $\mathscr{P}_{\text{test}}$-null. This is an intuitive idea, for if the support of $\mathscr{P}_{\text{test}}$ is not contained in the support of $\mathscr{P}$, then it is surely impossible that a reasonable prediction set/interval can be fit on some input $x$ that lies in the support of $\mathscr{P}_{\text{test}}$, but not in the support of $\mathscr{P}$. Under this assumption, we obtain the existence of the Radon-Nikodym derivative
$$
w(x)=\frac{\mathrm{d}\mathscr{P}_{\text{test}}}{\mathrm{d}\mathscr{P}}(x),
$$ called the likelihood ratio.

Given knowledge of the distributions $\mathscr{P}$ and $\mathscr{P}_{\text{test}}$ beforehand, the likelihood ratio can oftentimes be directly calculated via taking the ratio of the probability density functions of $\mathscr{P}_{\text{test}}$ and $\mathscr{P}$, respectively. This is sometimes infeasible, and this is the assumption we make in our conformal prediction algorithm, following the approach in [2]. We create feature-pairs $(X_i, C_i)$ of both training and testing data, where $C_i=1$ if $X_i$ is from the training data and $C_i=0$ if $X_i$ is from the testing data. We then train a multilayer perceptron on our feature-pairs using Keras/Tensorflow to approximate the probability
$$
\widehat{p}(x)=P(C=1\mid X=x).
$$ Observe now that
$$
w(x)=\frac{P(C=0)}{P(C=1)}\cdot \frac{P(C=1)P(X=x\mid C=1)}{P(C=0)P(X=x\mid C=0)}=\frac{P(C=0)}{P(C=1)}\cdot\frac{P(C=1\mid X=x)}{P(C=0\mid X=x)}
$$ by an application of Bayes' Theorem. Normalization constants turn out to be negligible in our later calculations, so we can directly take
$$
w(x)=\frac{P(C=1\mid X=x)}{P(C=0\mid X=x)}=\frac{\widehat{p}(x)}{1-\widehat{p}(x)}
$$ as the approximation for our likelihood function.

Using the approximated likelihood ratio, we now show the algorithm for performing conformalized quantile regression when there is a covariate shift. Given a user-chosen error rate $\alpha$, we train two gradient boosting models on a weighted quantile loss objective function on the training data $X_\text{train}$ to estimate the $\alpha/2$ and $1-\alpha/2$ quantiles of $Y\mid X=x$, for all $x\in X_{\text{train}}$. We denote these learned quantiles as $\widehat{t}_{\alpha/2}(x)$ and $\widehat{t}_{1-\alpha/2}(x)$, respectively. For a calibration point $(X_\text{cal}, Y_\text{cal})$, we define its score as
$$
s(X_\text{cal}, Y_\text{cal})=\max(\widehat{t}_{\alpha/2}(x)-Y_\text{cal}, Y_\text{cal}-\widehat{t}_{1-\alpha/2}(x)).
$$ Doing this for all $n$ calibration points, we obtain scores $s_1,\dots, s_n$, which we assume are already in sorted order. Define
$$
\widehat{q}(x)=\inf\left\{s_j:\sum_{i=1}^j p_i^w(x)\mathbb{I}_{\{s_i\leq s_j\}}\geq 1-\alpha \right\}
$$ where
$$
p_i^w(x)=\frac{w(X_i)}{\sum_{j=1}^n w(X_j)+w(x)},
$$ for $1\leq i\leq n$, with $X_1,\dots, X_n$ being the calibration data. We then obtain our prediction interval as
$$
C(x)=\left\{y: s(x,y)\leq \widehat{q}(x)\right\}=\left[\widehat{t}_{\alpha/2}(x)-\widehat{q}(x), \widehat{t}_{1-\alpha/2}(x)+\widehat{q}(x)\right],
$$ as desired. Note that these weights are actually dependent on $x$, so for every $x$ that we sample, we may observe a different width for each prediction interval. This makes sense, since we should be more/less confident based on the proximity of our input $x$ to our training/calibration data and is the reason that the weights $p_i^w$ are calculated in the manner above.

Sample output can be found in the Jupyter notebook.

## Sources
[1] Angelopoulos, A. N., & Bates, S. (2022). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv. https://arxiv.org/abs/2107.07511

[2] Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2020). Conformal prediction under covariate shift. arXiv. https://arxiv.org/abs/1904.06019
