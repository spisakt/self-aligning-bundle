---
title: "The self-aligning bundle: how matter with slow, coarse-grained dynamics comes to infer and learn"
exports:
  - format: docx
    output: self-aligning-bundle-draft.docx
  - format: pdf
    template: arxiv_nips
    output: self-aligning-bundle-draft.pdf
  - format: tex
    template: arxiv_nips
    output: latex/self-aligning-bundle-draft
abbreviations:
  FEP: Free Energy Principle
  VFE: Variational Free Energy
  NESS: Non-Equilibrium Steady State
  MZ: Mori–Zwanzig
  POV: Point of View
  FEP-ANN: Free-Energy-Minimizing Artificial Neural Network
---

+++ {"part": "abstract"}
Living and artificial systems that cope with changing environments come to reflect the regularities of these environments on two timescales: their states track what is happening now (perception), and their structure gradually absorbs what tends to happen (learning). Here we ask, what is a minimal physics that is sufficient for this. We start from a finite system, whose microstates follow a continuous-time Markov jump process and whose stationary state is an exponential tilt of an arbitrary positive reference ensemble. Normalizing this tilt over coarse, timescale-separated coordinates turns it - without any further assumption - into a prior, a likelihood, a posterior and an evidence; the conditional ensembles form a statistical bundle, whose vertical and horizontal centered scores are a prediction error and a contrastive Hebbian error. Taking a point of view - an exposed coordinate, its context and the structure that couples them - with bounded coordinates gives a sigmoid response, in which the fluctuations carry the uncertainty (the variance equals the gain). Perception and learning then admit an exact reading as variational free energy minimization on two timescales: the structure climbs its average log-evidence toward a ceiling set by its own model family (self-alignment), and it is separated from the environment by a path - but not a state - blanket. If a unit's renewal is caused by its context at the opening of a short window, networks of such units recover the learning rule of free-energy-minimizing attractor networks, including its temporal offset, the self-orthogonalization of the attractors and sequence learning. Preliminary simulations of jump processes illustrate these results. In the bigger picture, we suggest that life may be understood as a self-organized stacking of such timescales.
+++

(sec-introduction)=
## Introduction

- Living and artificial systems that cope with changing environments come to reflect the regularities of these environments on two timescales: their states track what is happening now - we call this *perception* -, and their structure gradually absorbs what tends to happen - we call this *learning*.
- We use these terms in a sense broad enough to cover a bacterium adjusting its tumbling, a cell collective regenerating a target shape, a brain recognizing a face, and a material that reshapes itself under repeated load.
- Many approaches formalize this: statistical accounts (perception as inference of hidden causes, learning as model fitting) {cite:p}`10.1016/j.tins.2004.10.007; 10.1111/j.2517-6161.1977.tb01600.x`, predictive coding {cite:p}`10.1038/4580; 10.3389/fnhum.2010.00215`, attractor networks and Boltzmann machines {cite:p}`10.1073/pnas.79.8.2554; 10.1207/s15516709cog0901_7`, description-length accounts {cite:p}`10.1016/0005-1098(78)90005-5; 10.1145/168304.168306; 10.48550/arXiv.2607.18433`, physical learning {cite:p}`10.1146/annurev-conmatphys-040821-113439; 10.1103/PhysRevX.11.021045; 10.3389/fncom.2017.00024` and basal cognition {cite:p}`10.1007/s10339-005-0016-8; 10.1371/journal.pcbi.1002739; 10.48550/arXiv.2602.08079`. These approaches however rarely say how the capacities they formalize arise from physical dynamics in the first place.
- The Free Energy Principle (FEP) makes perhaps the most general claim: self-organization to non-equilibrium steady states can itself be read as Bayesian inference, so that suitably persisting systems appear to minimize variational free energy (VFE), and thereby to perceive and learn {cite:p}`10.1016/j.physrep.2023.07.001; 10.1155/2012/937860; 10.1016/j.jtbi.2019.110089`.
- In its common formulation, internal and external states are separated by a Markov blanket and, *under suitable conditions*, internal dynamics minimize VFE. "*Under suitable conditions*", however, carries much of the explanatory burden, and it is here that the FEP received its most serious criticism: the required conditions may rarely hold {cite:p}`10.1016/j.plrev.2021.11.001; 10.3390/e23030293`, and the inferential reading may be projected onto systems, rather than found in them {cite:p}`10.1017/S0140525X21002351; 10.1007/s10539-021-09807-0`.
- Free-energy-minimizing attractor neural networks (FEP-ANNs) {cite:p}`10.1016/j.neucom.2026.133472` change the explanatory target. They compose simple units - each minimizing a local VFE - into attractor networks with sequence and continual learning, multi- and metastability and **approximately orthogonal attractors**, the latter with preliminary support from human neuroimaging {cite:p}`10.7554/eLife.98725`. Thus, instead of a path to the FEP for the most complex systems, we need only a plausible path to their building blocks.
- Here we use Occam's razor to identify the simplest plausible physical route to a minimal FEP-ANN unit. Whenever several structures could support the construction, we choose the least elaborate one; greater complexity is outsourced to the collective dynamics of units and to higher levels of organization.
- The central object of the construction, which we call a *self-aligning bundle*, is the geometry in which matter with slow, coarse-grained dynamics can start with a poor internal model of its environment, and improve it. Specifically, we show that:
  - normalization of a tilted reference ensemble over fast and slow coarse coordinates produces posteriors and evidences, without assigning beliefs to microstates,
  - for bounded coordinates, the leading interaction yields a sigmoid response (from boundedness alone), a prediction error and a contrastive Hebbian learning signal, as the two centered scores of the bundle,
  - fluctuations carry uncertainty at both timescales, and the slow dynamics separate learnable from unlearnable surprise,
  - no Markov blanket is assumed; a blanket is derived only between timescales, as a path (but not a state) blanket around the structure, while the split into an exposed coordinate and its context is a *point of view*, not a physical barrier,
  - in networks, the construction derives the learning rule of FEP-ANNs, including the temporal offset of its two factors - and this offset is what makes the attractors self-orthogonalize and capable of sequence learning.
- We illustrate the construction with a simulated finite-state jump process in which no inference or learning rule is coded (preliminary results).

(sec-main-results)=
## Main Results: the self-aligning unit

(sec-the-ansatz)=
### The Ansatz: a tilted reference ensemble and three timescales

We start from a substrate-independent but physically plausible formalism: a system whose stationary states maximize entropy relative to an arbitrary positive reference state, subject to the constraints imposed on it. Let $x$ be a microstate of a finite system - whose dynamics we take throughout to be a continuous-time Markov jump process, as for molecular conformations, chemical reaction networks or discrete spins (a continuous, diffusion picture would work too, but requires additional assumptions; see [Supplement S1](#app-s1)) - and $\rho_0$ a strictly positive reference density - e.g. a thermal equilibrium state, an unconstrained maximum-entropy distribution, or - equivalently - the stationary state under an earlier forcing/constraints. Any new perturbation of the stationary density $\rho_u$ (that is positive and sufficiently regular relative to $\rho_0$) can be written, without loss of generality, as a non-parametric exponential tilt {cite:p}`10.1214/aos/1176324311`:

$$
\rho_u=e^{\,u-K(u)}\rho_0,\qquad K(u)=\log\mathbb E_{\rho_0}[e^u],\tag{1}
$$

where the field $u$ (fixed up to an additive constant, $\mathbb E_{\rho_0}[u]=0$) represents the environmental perturbation.

Next, we consider coarse observables over these microstates, i.e. functions that pool many microstates into a single value (e.g. a population average $f(x)=\tfrac{1}{|G|}\sum_{i\in G}x_i$ over a group $G$ of microscopic degrees of freedom). From the vast number of possible coarse observables, we are interested in the slowest ones. Under plausible conditions (e.g. a suitable spectral structure), these give rise to a separation of timescales. We will show that, while a single gap suffices for an elementary form of perception,  a minimal adaptive unit needs three levels (separated by two gaps): microscopic variables $x$ relax first, retained mesoscopic coordinates $z=\pi(x)$ respond next, and macroscopic, *structural* coordinates $W=w(z)$ adapt much more slowly. The environment is assumed to vary on an intermediate timescale $\tau_u$ (or, more precisely, our unit is "sensitive" to intermediate external dynamics):

$$
\tau_x\ll\tau_z\ll\tau_u\ll\tau_W.\tag{2}
$$

In other words, the mesoscopic state tracks each environmental condition directly, while the structure “feels” them only on average.

(sec-normalization-turns-the-tilt)=
### Normalization turns the tilt into evidence

Coarse-graining the tilt $u$ (representing the external forcing) gives an effective field on the retained coordinates, $u_Z(z)=\log\mathbb E_{\rho_0}[e^u\mid z]$; apart from the Markovian reduction below, this is all the microstates contribute ([Supplement S2](#app-s2)). The tilted stationary law of the retained coordinates is:

$$
p_u(dz,dW)\propto e^{u_Z(z)}\,p_0(dz\mid W)\,p_0(dW).\tag{3}
$$

Normalizing it in its two natural ways gives two statements at once:

$$
p_u(dz\mid W)=\frac{e^{u_Z(z)}\,p_0(dz\mid W)}{Z_u(W)},\qquad p_u(dW)\propto Z_u(W)\,p_0(dW),\qquad Z_u(W)=\int e^{u_Z(z)}\,p_0(dz\mid W).\tag{4}
$$

The probabilistic reading is direct: $p_0(dz\mid W)$ is a **prior** selected by the structure, $e^{u_Z}$ an unnormalized **likelihood**, $p_u(dz\mid W)$ the **posterior** and $Z_u(W)$ its **evidence**. Note that no beliefs have been assigned to microstates: the Bayesian structure emerges simply from the normalization of a coarse stationary family. Crucially, the structure W feels the environment only through the evidence of the conditional model it indexes.

Normalization also fixes the form of all tangents. Along any smooth curve $p_t$ of normalized densities, differentiating the constraint $\int p_t=1$ gives $\mathbb E_{p_t}\big[\tfrac{d}{dt}\log p_t\big]=0$, meaning that *every score is centered.* Eq. (1) supplies such curves: perturbing a density $p$ in the direction of an observable $f$ (i.e. tilting it as $p_t\propto e^{tf}p$) gives $\log p_t=tf-K(tf)+\log p$, so the score is $f-K'(0)=f-\mathbb E_p[f]$. In other words, it is the log-normalizer $K$ that subtracts the mean, and the tangent is “actual minus expected”:

$$
\tfrac{d}{dt}\log p_t\Big|_{t=0}=f-\mathbb E_p[f].\tag{5a}
$$

The Fisher inner product of two such tangents is their covariance:

$$
g_p(f,h)=\operatorname{Cov}_p(f,h).\tag{5b}
$$

Conversely, every tangent has this form (take $f$ to be its own score), so that the tangent space consists of the centered observables. As we will see, the centered tangents become the errors of the minimal unit (with $f=\mu$, the prediction error; with $f=s\mu$, the contrastive learning signal), and their covariance the metric that links the unit's fluctuations to its responses.

(sec-the-self-aligning-bundle)=
### The self-aligning bundle: fibers, section and two directions

We make one dynamical assumption, namely *adiabatic conditional relaxation*: at each frozen value of the slow coordinates, the fast coordinates follow an effectively Markovian dynamics with a unique stationary conditional, and relative entropy to that conditional contracts on a timescale short compared to the motion of the structure (for a justification via Mori–Zwanzig projection and spectral-gap conditions, see [Supplement S1](#app-s1)). The second law, in its relative-entropy form, then drives the fast law toward $p_u(dz\mid W)$ and - one level up - the structural law toward $p_u(dW)$.

This organization gives rise to a specific information geometric description, as shown on [Figure 1](#fig-theory). The conditional distributions of the fast coordinates at a fixed structure $W$ form a *fiber* $\mathcal E_W$, and the fibers form a local statistical bundle over the space of structures {cite:p}`10.1214/aos/1176324311`. The stationary conditionals select a distinguished *section* $\sigma_u(W)=p_u(\cdot\mid W)$, and near it the tangent space splits into vertical and horizontal parts:

$$
T_{\sigma_u(W)}\mathcal E=V\oplus H,\qquad V=\ker d\pi,\qquad H=d\sigma_u\big(T_W\mathcal W\big).\tag{6}
$$

Vertical tangents change the fast conditional at fixed structure; horizontal tangents change the structure while staying on the section. Fast relaxation moves vertically toward the section - this will be perception -, and slow adaptation moves along it - this will be learning. Since both motions occur within normalized families, both tangents are centered scores (eq. (5a)). What remains, is to choose coordinates in which the two scores are explicit.

```{figure} fig/fig1_theory.png
:name: fig-theory
:width: 100%

**From a general, continuous-time discrete-state jump process to perception and learning.** **A**: Microstates (gray) and their transitions. Coarse observables pool overlapping groups of microstates: the fast exposed coordinate $\mu$ and context $s$, and a slow structural coordinate $W$. Their relaxation times fall into separated bands (bottom). **B**: At every structure, the conditional laws of the fast coordinates form a fiber; the environment selects the stationary section $\sigma_u$, and the unit's own statistics the resting section $\sigma_0$. Perception relaxes the fast state along the fiber onto $\sigma_u$; learning moves the structure along the section. **C**: Perturbing a normalized density $p$ along an observable $f$ stays within the simplex only once the expected value is subtracted. For the exposed coordinate, this centered tangent is the prediction error. **D**: The coupling $W$ and the correlation $C$ it predicts are dual coordinates of the log-normalizer $\psi$, related by its slope; its curvature is the Fisher metric $g$.
```

(sec-a-point-of-view)=
### A point of view and its minimal model

The coordinates within a fiber are fixed in three steps, of which only one is a genuine choice. The split into fast $z$ and slow $W$ is already given by the timescales (eq. (2)). Within the fast sector, we then choose one coordinate to follow - a *point of view** **(**P**OV**)** **w**hi**ch we* defined below in detail.  The POV is the only choice we make at this point. When combining self-aligning units into a FEP-ANN, it will turn out that a reasonable minimal choice of a POV coordinate is whatever is exposed to other units in the network. Finally, the structural coordinate is not chosen at all: $W$ is defined as the coefficient of the leading context–exposed interaction (eq. (7b)), which makes it the natural parameter of the unit's model, with the expected correlation $C=\mathbb E[s\mu]$ as its Legendre dual.

We take all retained mesoscale coordinates to be bounded - as in any finite physical system - and rescale them to $[-1,1]$. Note that this is not only realistic, but also technically required: for a coordinate constrained only by its range, the least structured family with a constrained mean is a linear tilt $e^{v\mu}$, which exists for every value of the field only if the range is bounded.

**Next - as a key move - we select a single distinguished mesoscale coordinate:** a *point of view* (POV) on the unit, which we refer to as the *exposed coordinate* $\mu\in[-1,1]$. We follow its conditional response to the remaining coordinates, the *context* $s\in[-1,1]^n$.

The slow coordinates $W$ that shape this response constitute the unit's *structure*. Relative to a point of view, every influence on $\mu$ is either mediated by $W$ - and belongs to the unit's model -, or not - and belongs to the tilt (i.e. to the environmental field $u$ of eq. (1), which reweights the reference ensemble from outside the unit's model).

Note that, while our notation is suggestive of the FEP convention ($s$ for sensory and $\mu$ for internal states), no blanket is assumed between them: the context is a set of distinguished coordinates, not a physical barrier - which is why we deliberately avoid calling it a boundary (which blanket nonetheless follows from the construction, is shown in [§ A path blanket between timescales](#sec-a-path-blanket-between)). Since any bounded coordinate can be chosen, a network is naturally a collection of points of view, each coordinate exposed from its own and part of the context from the others'.

Importantly, the point of view induces an exact decomposition of the effective field $u_Z$ into a part that reaches $\mu$ only through the context, a part that acts on $\mu$ directly, and their interaction:

$$
u_Z(s,\mu)=u_S(s)+u_\mu(\mu)+u_{s\mu}(s,\mu).\tag{7a}
$$

Here we make the **main simplification** of the construction (assumption 6): for each direction of the bundle, we keep only the lowest-order sufficient statistic - $\mu$ for the vertical and $s\mu$ for the horizontal direction. In other words, the unit constrains only the mean of its exposed coordinate and its correlation with the context, for which the least structured (maximum-entropy) family is exactly linear and bilinear (for the motivation and the limits of this choice, see [Supplement S6](#app-s6)). The context tilt $u_S$ remains arbitrary, the environment adds a direct field $v$ on $\mu$, and the structure enters through the leading cross-term:

$$
u_Z(s,\mu)=u_S(s)+v\,\mu,\qquad W\cdot s\,\mu,\quad W\in\mathbb R^n.\tag{7b}
$$

Context-only forcing ($v=0$) is the special case that corresponds to the Markov blanket of standard FEP formulations, in which the context alone screens the exposed coordinate from the environment; the construction includes it, but does not require it ([§ A path blanket between timescales](#sec-a-path-blanket-between)) (for higher-order terms and degenerate cases, see [Supplement S6](#app-s6)). At rest and uncoupled, $\mu$ is constrained only by its range, so its maximum-entropy law is uniform {cite:p}`10.1103/PhysRev.106.620`. With the bilinear interaction, the unit's model is:

$$
p_0(ds,d\mu\mid W)=e^{\,W\cdot s\,\mu-\psi(W)}\,r_0(ds)\,\frac{d\mu}{2},\qquad \psi(W)=\log\iint e^{\,W\cdot s\,\mu}\,r_0(ds)\,\frac{d\mu}{2},\tag{8}
$$

where $r_0(ds)$ is the resting context law. Conditioning on $s$ gives the unit's **prediction** for its exposed coordinate:

$$
p_0(d\mu\mid s,W)=e^{\,\theta\mu-A(\theta)}\,\frac{d\mu}{2},\qquad \theta=W\cdot s,\qquad A(\theta)=\log\frac{\sinh\theta}{\theta},\tag{9}
$$

with mean and variance:

$$
L(\theta)=A'(\theta)=\coth\theta-\frac{1}{\theta},\qquad A''(\theta)=\operatorname{Var}(\mu\mid s,W).\tag{10}
$$

The mean map $L$ is the Langevin function of classical paramagnetism (not to be mistaken with Langevin dynamics). Crucially, no activation function is chosen by hand here: $L$ is the Legendre map from the natural to the expectation coordinate of the bounded family, and it is sigmoid simply because $\mu$ is bounded. A binary $\mu$ gives $\tanh$, and the continuous Bernoulli of FEP-ANNs is exactly this family, after the map $[0,1]\to[-1,1]$.

**Fluctuations and precision.** Since $L=A'$, eq. (10) also states that the variance of $\mu$ equals the slope of its mean response, $L'(\theta)=\operatorname{Var}(\mu\mid s,W)$ - the fluctuation–dissipation relation of the exponential family {cite:p}`10.1088/0034-4885/29/1/306`. Thus, a unit is most responsive where it is most uncertain, and the precision of its prediction, $1/A''(\theta)$, is not a separate parameter, but its inverse gain - a physically measurable quantity. As we show next, the fluctuations of $\mu$ then sample the posterior, so that the unit's uncertainty is expressed both in the size of its fluctuations and in its gain.

In these coordinates, both directions of the bundle carry a Legendre pair, a metric and a centered score:

| Direction | Natural coordinate | Expectation coordinate | Metric | Centered score |
| --- | --- | --- | --- | --- |
| vertical (fast) | $\lambda$ (current conditional $q\propto e^{\lambda\mu}$) | $m=L(\lambda)$ | $A''(\lambda)$ | $\mu-L(\lambda)$ |
| horizontal (slow) | $W$ | $C(W)=\nabla\psi=\mathbb E_{p_0}[s\mu]$ | $g(W)=\nabla^2\psi=\operatorname{Cov}_{p_0}(s\mu)$ | $s\mu-C(W)$ |

The unit's prediction sits at $\lambda=\theta$ and - as shown next - the stationary section at $\lambda=\theta+v$.

(sec-perception-as-vertical-relaxation)=
### Perception as vertical relaxation

Under leading-order forcing, the stationary conditional of $\mu$ is the unit's prediction tilted by the direct field:

$$
p_v(d\mu\mid s,W)=\frac{e^{v\mu}\,p_0(d\mu\mid s,W)}{\mathbb E_{p_0(\cdot\mid s,W)}[e^{v\mu}]}=e^{(\theta+v)\mu-A(\theta+v)}\,\frac{d\mu}{2}.\tag{11}
$$

By eq. (4), this is a Bayesian posterior, with the prediction as prior and $e^{v\mu}$ as likelihood. In other words, prediction and evidence simply add in the natural coordinate: $\theta$ carries what the structure expects, and $v$ what the environment adds. The corresponding log-evidence is $\log Z(s,v;W)=A(W\cdot s+v)-\psi(W)$. Up to sign and temperature, $-\log Z$ is the thermodynamic free energy of the fast sector, and it turns into a variational free energy through the Gibbs variational principle: the free energy of a stationary ensemble is the minimum, over all trial distributions $q$, of their mean energy minus their entropy - here, with the negative log of the unnormalized joint, $-\log\big(e^{v\mu}p_0(s,\mu\mid W)\big)$, playing the role of energy. Evaluated for an arbitrary, not yet relaxed $q$, this functional is exactly the variational free energy of variational Bayes and the FEP. For any trial conditional $q(d\mu)$, the variational free energy decomposes as:

$$
\mathcal F(q;s,v,W)=\int q(d\mu)\log\frac{q(\mu)}{e^{v\mu}\,p_0(s,\mu\mid W)}=D_{\mathrm{KL}}\big(q\,\Vert\,p_v(\cdot\mid s,W)\big)-\log Z(s,v;W).\tag{12}
$$

Adiabatic relaxation contracts exactly the divergence in this decomposition, so - at fixed structure - fast relaxation *is* free-energy descent. Near the posterior, this is (to leading order) first-order relaxation of the mean $m=\mathbb E_q[\mu]$, i.e. the unit's activation:

$$
\dot m=-\gamma\big(m-L(W\cdot s+v)\big)+\omega_m,\qquad\gamma>0,\tag{13}
$$

where $\omega_m$ collects finite-size fluctuations (it may be omitted in a deterministic artificial unit).

**Fluctuations carry uncertainty.** The exposed coordinate does not sit at its posterior mean; it fluctuates around it with the posterior variance $A''(\theta+v)$, i.e. each sample of $\mu$ is a sample from the posterior - as in sampling-based accounts of neural computation {cite:p}`10.1016/j.tics.2010.01.003; 10.1371/journal.pcbi.1002211`. By the fluctuation–dissipation relation introduced with eq. (10), the same number is the gain of the unit to the direct field:

$$
\frac{\partial\,\mathbb E[\mu\mid s,v,W]}{\partial v}=L'(\theta+v)=\operatorname{Var}(\mu\mid s,v,W).\tag{14}
$$

Thus, the uncertainty of the posterior is physically expressed in two ways: in the size of the unit's fluctuations and in its gain.

Note that two errors are in play here. The *perceptual error* $m-L(\theta+v)$ is removed by perception. The *prediction error* - the vertical score of the unit's own model -

$$
\varepsilon_\mu=\partial_\theta\log p_0(\mu\mid s,W)=\mu-L(\theta)\tag{15}
$$

has posterior mean $L(\theta+v)-L(\theta)$, which vanishes only without direct evidence. This error can not be removed by perception, only by learning. Its variance under the model, $A''(\theta)$, is the metric of the vertical direction, and its inverse is the precision of the unit's prediction.

(sec-learning-as-horizontal-motion)=
### Learning as horizontal motion

Learning concerns an environment that changes. Suppose it presents a sequence of conditions $k$ - each a context tilt $u_S^{(k)}$ together with a direct field $v_k$ -, held for a time $\tau_u$ and recurring with stationary frequencies $r_k$ that do not depend on $W$. During one presentation, the structure feels the environment only through its evidence $\log Z_k(W)$ (eq. (4)). Since it can not follow individual presentations, it responds to their average: by the averaging principle {cite:p}`10.1007/978-0-387-73829-1`, its effective slow potential is:

$$
\ell(W)=\sum_k r_k\log Z_k(W)+\log p_0(W)=\mathbb E_r\big[A(W\cdot s+v)\big]-\psi(W)+\log p_0(W),\tag{16}
$$

where the second form holds for clamped presentations, with $r$ the distribution of pairs $(s,v)$ (for soft presentations, see [Supplement S3](#app-s3)). Note that, apart from the prior, $\ell$ is minus the presentation-averaged minimum of the free energy (eq. (12)), i.e. the slow potential is the presentation-average of the fast objective.

Differentiating the unit's model along the structure gives the second centered score, the *horizontal score*:

$$
\varepsilon_W=\partial_W\log p_0(s,\mu\mid W)=s\mu-C(W),\tag{17}
$$

i.e. the observed context–exposed correlation minus the correlation the resting structure generates. Averaged over the driven fast state and over presentations, it is the evidence gradient:

$$
\nabla_W\ell=C_{\rm env}(W)-C(W)+\nabla_W\log p_0(W),\qquad C_{\rm env}(W)=\mathbb E_r\big[s\,L(W\cdot s+v)\big].\tag{18}
$$

This has the familiar form of contrastive learning, with a positive (“Hebbian”) and a negative (“anti-Hebbian”) phase. Crucially, however, the positive phase $C_{\rm env}$ contains every field acting on the unit, while the negative phase $C$ contains only its own structure. In other words, fields that the unit does not mediate enter the observed term, and not the predicted one.

**Registering the score.** In a finite-state jump process, coordinates change one at a time, and a learning event occurs only when the exposed coordinate itself jumps - its *renewal*; jumps of the context alone register nothing. We assume that a renewal is *caused* by the context at the opening of a short window - $\mu^+$ is drawn from its conditional given $s$, independently of the other coordinates -, while it is *registered* together with the context $s^+$ at the end of the window, by which time some context coordinates may already have changed silently (assumption 7, *causal renewal*). This is more realistic than an instantaneous update, as any physical response has a finite latency ([Supplement S1](#app-s1)), and it is exactly what is needed to recover the temporal offset of the learning rule of FEP-ANNs {cite:p}`10.1016/j.neucom.2026.133472` ([§ Composition and the learning rule](#sec-composition-and-the-learning)). For a clamped context, $s^+=s$ within a presentation, and the registered statistic is simply the equal-time score of eq. (17), apart from renewals right after a switch ([Supplement S3](#app-s3)).

By assumption 5, applied one level up, the structural law relaxes toward $e^{\ell}$. Up to a circulation that preserves this law (left free by the construction, see [Supplement S3](#app-s3)), the learning law - in the small-step limit of the structural kinetics, a diffusion - is:

$$
\dot W=\Gamma\,\nabla_W\ell(W)+\omega_W,\tag{19}
$$

with $\Gamma$ a positive mobility and $\omega_W$ the structural noise. The structure therefore relaxes toward $p(W)\propto e^{\ell(W)}$, a Gibbs posterior over couplings. Structural noise plays, one level up, the role that fluctuations play for the fast state: it keeps the uncertainty about the structure represented, it breaks the symmetry of the blind state at the start of learning, and it sets the resolution at which learned structure is encoded. In the low-noise limit, learning becomes deterministic ascent of $\ell$, i.e. MAP estimation.

Note that the actual microstate mobility only affects the trajectories of learning; the fixed points and the stationary law are the same for every positive $\Gamma$. One choice is however distinguished by the geometry. With the natural-gradient mobility $\Gamma=\kappa\,g(W)^{-1}$, the modeled correlations relax linearly:

$$
\dot C=g\,\dot W=\kappa\,\big(C_{\rm env}-C\big),\tag{20}
$$

i.e. in expectation coordinates, learning is first-order relaxation toward the correlations that the environment sustains - the slow counterpart of eq. (13). As for the fast coordinate (eq. (14)), $g$ is both the covariance of $s\mu$ and its susceptibility to the couplings.

(sec-self-alignment)=
### Self-alignment: the alignment gap and learnable surprise

We measure the unit's mismatch with its environment by its average surprisal, the *alignment gap*:

$$
\mathcal A(W)=-\sum_k r_k\log Z_k(W)+c=\sum_k r_k\min_q\mathcal F_k+c,\tag{21}
$$

with $c$ a $W$-independent constant. With context-only forcing, $c$ can be chosen so that $\mathcal A(W)=D_{\mathrm{KL}}\big(r\,\Vert\,p_0(\cdot\mid W)\big)$. Without the prior, the gradient part of learning decreases the gap monotonically, since $\dot{\mathcal A}=-\nabla\ell^\top\Gamma\,\nabla\ell\le0$ for any positive mobility. Three properties make the bundle *self-aligning*.

**Starting blind.** At $W=0$ the unit predicts nothing ($\theta=0$), and - for a prior centered at zero - the learning drift is $\Gamma\,\mathbb E_r[s\,L(v)]$. Whenever the context is correlated with the direct evidence, learning starts at once: the unit learns to predict its own input from its context. A fully shielded unit ($v=0$) has the blind state as a generically unstable fixed point, which it leaves along the direction in which the environment excites the context most strongly, akin to Hebbian incremental principal-component learning {cite:p}`10.1007/BF00275687` ([Supplement S4](#app-s4)).

**The ceiling.** At a fixed point, $C(W^\ast)=C_{\rm env}(W^\ast)$: the correlations generated by the resting structure match those that the environment sustains through the unit's own posterior - a self-consistent moment matching ([Supplement S5](#app-s5)). The achievable alignment is capped by the model family and by physical constraints on $W$. Thus, alignment is partial at every stage; the geometry only guarantees a tendency to improve. As a toy example, consider the simplest case, in which $W$ indexes only two discrete structures, $W_1$ and $W_2$. The stationary law then weighs them by $p_0(W_i)\,e^{\ell(W_i)}$, i.e. the dynamics select the structure that is better aligned with the environment - even if the alignment is poor for both.[^ceiling]

[^ceiling]: The present derivation says nothing about how high the ceiling is and in most systems that suffice the definition of a self-aligning bundle, the ceiling is likely so low that it is hard to interpret it as genuine learning. In the bigger picture, we assume that evolution - by exploring substrates, coordinates and couplings - keeps pushing this structural ceiling ever higher (see Discussion).

**Learnable and unlearnable surprise.** The stationary law of the structure is itself the minimizer of a slow free energy:

$$
\mathfrak F[Q]=\mathbb E_Q\big[\mathcal A(W)\big]+D_{\mathrm{KL}}\big(Q\,\Vert\,p_0\big),\qquad Q^\ast\propto p_0\,e^{-\mathcal A}\propto e^{\ell}.\tag{22}
$$

Its two terms split the unit's surprise into an unlearnable part - the residual gap $\mathbb E_{Q^\ast}[\mathcal A]$ that no structure in the family removes - and a learnable part, $D_{\mathrm{KL}}(Q^\ast\Vert p_0)$, which is (by the bits-back argument) the description length of what the unit has learned {cite:p}`10.1145/168304.168306` ([Supplement S7](#app-s7)).

(sec-a-path-blanket-between)=
### A path blanket between timescales

Our construction does not assume a Markov blanket, but it does derive one - between timescales. We use the distinction of {cite:t}`10.48550/arXiv.2607.13391` between a *state blanket*, which screens two sets of variables at a single time, and a *path blanket*, which screens their histories; neither implies the other. Since the environment reaches the structure only through the fast coordinates (assumption 4), the fast history screens the history of the structure from that of the environment $\eta$:

$$
W_{[0,T]}\perp\!\!\!\perp\eta_{[0,T]}\mid(s,\mu)_{[0,T]},\qquad\text{while generically}\qquad W_t\not\perp\!\!\!\perp\eta_t\mid(s_t,\mu_t).\tag{22a}
$$

The state version fails for two reasons: the structure integrates past presentations (the mixture obstruction of {cite:t}`10.48550/arXiv.2607.13391`), and conditioning on $\mu_t$ - which depends on both $W_t$ and $\eta_t$ - couples the two (a collider effect). Importantly, this failure is expected: the information that the structure carries about the environment beyond the current fast state is exactly what it has learned. In this sense, learning can be characterized by the presence of a path blanket in the absence of a state blanket. Note that no blanket is required at the fast level: the environment may act on $\mu$ directly - and must, if the unit is to learn more than the statistics of its context. The same distinction reappears in networks: a renewed state $\mu^+$ is screened by the *preceding* context $s$, not by $s^+$, so the learning statistic is a two-time, path-level object (eq. (23)).

(sec-interim-summary)=
### Interim summary: two-scale free-energy minimization

To sum up, our construction requires the following assumptions (for their scope, see [Supplement S1](#app-s1)):

1. **Reference ensemble:** a continuous-time Markov jump process on a finite state space, with a positive reference ensemble whose stationary densities under different external forcings/constraints form a regular family of constrained maximum-entropy densities - each of which then takes the general tilt form of eq. (1),
2. **Double timescale separation** among the slowest observables (e.g. via spectral gaps), which defines the mesoscale coordinates $z$ and the structural coordinates $W$; the environment varies in between, with presentation statistics that do not depend on $W$ (eq. (2)),
3. **Bounded coordinates** (an unbounded coordinate would require further constraints, e.g. a fixed variance),
4. **Mesoscale sufficiency:** the environmental forcing visible to the structural observables is fully captured by the mesoscale observables, i.e. the environment reaches $W$ only through $z$,
5. **Adiabatic conditional relaxation:** timescale separation removes memory, so that each level relaxes toward its stationary conditional by relative-entropy contraction (“pure entropic dynamics”),
6. **Leading order:** for each direction of the bundle, only the lowest-order sufficient statistic is kept (eq. (7b)), with the maximum-entropy resting law of $\mu$.
7. **Causal renewal:** a learning event is caused by the context at the opening of a renewal window and registered with the context at its end - harmless for a clamped context, and consequential only when the context changes on the fast timescale, as in networks.

Note that the point of view is a modeling choice rather than an assumption. Slow diffusive kinetics are not needed as a separate assumption either: the stationary law of the structure follows from assumptions 2 and 5, and the diffusion of eq. (19) is simply the small-step limit of the structural kinetics. Assumption 7 does not affect the single-unit results below, but fixes the timing of the learning rule in networks.

**Proposition (Two-scale free-energy minimization).** Under assumptions 1–6, a unit with exposed coordinate $\mu$, context $s$ and structure $W$ has the following properties:

1. **Perception.** At fixed structure, the mesoscale conditional descends the variational free energy (eq. (12)) toward the posterior (eq. (11)).
2. **Learning.** Its structure descends the presentation-averaged minimum of the free energy, i.e. it ascends the average log-evidence $\ell(W)$ (eq. (16)).
3. **Self-alignment.** Along the gradient part of learning, the alignment gap decreases monotonically, until the self-consistent moment matching $C(W^\ast)=C_{\rm env}(W^\ast)$ is reached. The residual gap - the ceiling of alignment - is set by the model family and by the physical constraints $x$ puts on $W$; the construction guarantees only the tendency to improve, and says nothing about the height of this ceiling. Finally, the stationary law of the structure splits the unit's surprise into a learnable and an unlearnable part.
4. **Uncertainty.** At both timescales, stationary fluctuations sample the corresponding posterior; the variance of the fast fluctuations equals the unit's gain.

The division of labor is that of expectation–maximization {cite:p}`10.1016/0893-6080(95)00003-8`: perception computes the posterior, and learning climbs the evidence given that posterior. Note that no physical variable needs to represent a full probability distribution: a retained coordinate needs only to track a sufficient mean of a locally closed conditional family.

(sec-from-units-to-networks)=
## From units to networks: relation to FEP-ANNs

(sec-correspondence)=
### Correspondence

The minimal unit maps one-to-one onto a single node of an FEP-ANN {cite:p}`10.1016/j.neucom.2026.133472`:

| Minimal unit | Role in the construction | FEP-ANN node |
| --- | --- | --- |
| $\mu$ | exposed coordinate | node state |
| $s$ | context | states of the other nodes |
| $W$ | structure | incoming coupling vector |
| $v$ | direct field | external input (any input not passing through the node's couplings) |
| $L(\theta)$ | Langevin mean map | activation (continuous Bernoulli) |
| $\mu-L(\theta)$ | prediction error | prediction error |
| $s\mu$ and $C(W)$ | positive and negative phase | Hebbian and anti-Hebbian terms |
| $\mathcal F(q;s,v,W)$ | local VFE | node-wise VFE |

(sec-composition-and-the-learning)=
### Composition and the learning rule

After deriving perception and learning from a POV on timescale-separated observables over the microstates of a dynamical system, we now investigate the collective dynamics of multiple such units combined into a network, as discussed in {cite:t}`10.1016/j.neucom.2026.133472`. The key to connecting the present derivation to FEP-ANNs is that one unit's exposed coordinate must become part of other units' contexts. In this setup, a network node's context is not clamped by the environment, but produced by the other units, which change on the same fast timescale as $\mu$. Here, the renewal window introduced with the learning law ([§ Learning as horizontal motion](#sec-learning-as-horizontal-motion), assumption 7) becomes consequential: by the time a unit's renewal is registered, parts of its context - the exposed coordinates of other units - have typically been renewed already. As before, we write $s,\mu$ for the state at the opening of a window and $s^+,\mu^+$ for the state at its end; the synchronous network simulations below are the limiting case, in which every coordinate is renewed in every window. Thus, a unit forms its prediction from what it has already seen, $\theta=W\cdot s$, while its renewed state samples the posterior, $\mu^+\sim p_v(\cdot\mid s,W)$. The horizontal score - a statistic of the joint state - then separates into two components:

$$
s^+\mu^+-C(W)=s^+\big(\mu^+-L(W\cdot s)\big)+\big(s^+L(W\cdot s)-C(W)\big).\tag{23}
$$

Since $s^+$ and $\mu^+$ are drawn independently given the state at the opening of the window, the second component vanishes on resting statistics; under drive, it measures how the context departs from its resting law, which the context units model themselves. Each node keeps the first component, which gives the local learning rule:

$$
\Delta W\propto s^+\big(\mu^+-L(W\cdot s)\big),\tag{24}
$$

i.e. the error of the renewed state against the prediction from the *preceding* context, paired with the *renewed* context. This is exactly the rule of {cite:t}`10.1016/j.neucom.2026.133472`, and the construction accounts for two of its features, that were previously set empirically. First, fields not mediated by a node's couplings must enter the observed term and not the predicted one: from the node's point of view, such fields are tilt. Second, causality fixes the time of the prediction, and the joint statistic the time of the Hebbian factor. This temporal offset underlies the reading of the rule as a generalization of spike-timing-dependent plasticity (STDP). For a clamped context ($s^+=s$), the rule reduces to the gradient of the pseudo-log-likelihood ([Supplement S8](#app-s8)).

(sec-self-orthogonalization)=
### Self-orthogonalization

The temporal offset is what organizes the learned attractors. Let the environment present patterns $\xi_k$ in random order, each for many renewals, without resetting the fast state. Most renewals occur after the network has settled ($s^+\approx s$), but a fraction $f$ follows a switch: the preceding state still reflects the previous pattern, while the renewed state already reflects the new one. In the linear regime (states $a\xi$, $L(h)\approx h/3$), with $\bar m$ the mean pattern, $\Sigma=\mathbb E[\xi\xi^\top]$ and $\Sigma_c=\Sigma-\bar m\bar m^\top$, the fixed point on the span of the patterns is:

$$
W^\ast=3\,\Sigma\,\big[(1-f)\,\Sigma_c+\bar m\bar m^\top\big]^{-1}\tag{25}
$$

([Supplement S9](#app-s9)). The key intuition is that a prediction formed from a random predecessor predicts what the patterns share, and the error keeps what distinguishes the present one. Thus, the shared direction keeps gain 3, while the directions in which the patterns differ gain up to $3/(1-f)$, and the attractors inherit the geometry of the patterns with their common part suppressed - i.e. they become **approximately orthogonal**. The timing of the two factors is crucial here: pairing the error with the preceding context (as the pseudo-likelihood gradient does) reverses the sign of the switch term, so correlated patterns merge; and resetting the state between presentations removes the contrast.

(sec-collective-phenomena)=
### Collective phenomena

Networks of such units display a much richer behavior than single units. Sequential presentations lead to asymmetric couplings - since the prediction formed from the preceding pattern is then always the same one - and to non-equilibrium steady-state dynamics, supporting sequence learning, continual learning and multi- and metastability {cite:p}`10.1016/j.neucom.2026.133472`. The antisymmetric part of the couplings generates solenoidal flow, the network counterpart of the circulation left free at the unit level. Finally, if a collective mode becomes slow, it is promoted to the next coarse level, where evidence at one level acts as field at the next ([Supplement S2](#app-s2)), which gives the network a hierarchical message-passing reading.

(sec-in-silico-demonstrations-preliminary)=
## In silico demonstrations (preliminary)

(sec-simulation-1)=
### Simulation 1: a single unit as a finite jump process

- **Setup.** A finite-state, continuous-time jump process whose rates obey local detailed balance. The exposed coordinate in our initial simulation has 21 levels in $[-1,1]$, each carried by the same number of microstates; there is one context coordinate, a structural coordinate on a grid, and two alternating presentations.
- **Mechanism.** The exposed coordinate moves only during brief encounters with its context, as is natural for contact-mediated (e.g. chemical) interactions. No inference or learning rule is written into the rates.
- **Timescales.** The relaxation spectrum falls into three bands, separated by factors of 14 and 29 ([Figure 2A](#fig-unit)). The separation is set by the rate classes: microscopic hops and the closing of encounter windows are fast, encounters and context moves are mesoscale, and structural steps are attempted only at the closing of a small fraction (2%) of encounters.
- **Perception.** As predicted by theory, at fixed structure the stationary law of $\mu$ is the discretized continuous Bernoulli (the activation function we assumed in {cite:t}`10.1016/j.neucom.2026.133472`), its mean follows the Langevin function, and its variance equals its gain ([Figure 2B-C](#fig-unit)).
- **Learning.** Starting from the blind state, the structure climbs the average log-evidence along the learning law, and its stationary law matches the posterior over couplings, $p_0(W)\,e^{\beta\ell(W)}$ ([Figure 2D](#fig-unit)).

```{figure} fig/fig2_single_unit.png
:name: fig-unit
:width: 100%

**A single unit as a jump process.** **A**: The relaxation rates of the process fall into three separated bands: microscopic hops (gray), the mesoscopic exposed and context coordinates (blue), and the structure (purple). **B**: Since every level of $\mu$ carries the same number of microstates, the stationary law at a fixed field (bars) is the continuous Bernoulli (lines): uniform at zero field, exponentially tilted otherwise. **C**: Its mean follows the Langevin function; the inset shows that the variance of the fluctuations equals the gain. **D**: Starting blind, the structure learns: sample paths (thin), their mean (thick) and the learning law (dashed, with its spread in gray) rise toward the stationary law on the right, which is the posterior over couplings, $p_0(W)\,e^{\beta\ell(W)}$.
```

(sec-simulation-2)=
### Simulation 2: networks of jump-process units

- **Setup.** Networks of 64 units, one per pixel of $8\times8$ images, as in {cite:t}`10.1016/j.neucom.2026.133472`. All units renew together at each step, each from its conditional given the preceding network state and an observation of the presented pattern. The structure follows the rule of eq. (24), in its low-noise limit.
- **Self-orthogonalization.** Trained with the ten digits of {cite:t}`10.1016/j.neucom.2026.133472` in random order (pairwise deviation from orthogonality: 23.3° on average), the networks reproduced the results in {cite:t}`10.1016/j.neucom.2026.133472` ([Figure 3B-C](#fig-networks)).
- **Controls.** Pairing the error with the preceding context merged the digits into a single attractor in five of six runs. Resetting the state between presentations left 3–5 attractors, with a deviation of 29–42° and equal gains (0.99).
- **Sequence learning.** With three digits presented in cyclic order, the couplings became asymmetric (antisymmetric-to-symmetric norm 0.69, against 0.23 for random order), and the network - released without input - replayed the sequence forward (17 forward and no backward transitions in 60 renewals; [Figure 3A](#fig-networks)), reproducing the results in {cite:t}`10.1016/j.neucom.2026.133472`.

```{figure} fig/fig3_networks.png
:name: fig-networks
:width: 100%

**Networks of jump-process units learn sequences and self-orthogonalize.** **A**: Three digits presented in cyclic order. Released without input, the network replays them forward; the overlap of each state with the three digits shows the cycle, which is absent after random-order presentation. **B**: The ten digits and the attractors reached from them. Digits 0 and 4 share one attractor with opposite signs ($a_1$ and $-a_1$), i.e. evidence for or against the same pattern. **C**: Pairwise deviation of the distinct attractors from orthogonality in six runs. With the network rule, the attractors are more orthogonal than the digits (dashed line); pairing the error with the preceding context merges them, and resetting the state between presentations leaves fewer, less orthogonal attractors.
```

(sec-discussion)=
## Discussion

(sec-what-the-construction-shows)=
### What the construction shows

- Here we aimed to give a sufficient physical route to a minimal system that realizes the shared core of perception and learning, and can serve as a unit in the FEP-ANNs of {cite:t}`10.1016/j.neucom.2026.133472`, without having to assume the FEP itself.
- We showed that a normalized reference ensemble, an environmental tilt, coarse-graining, bounded coordinates and timescale separation are enough to produce evidences, posteriors, prediction errors and a local contrastive learning rule - without assigning beliefs to microstates.
- Fluctuations carry the uncertainty of both posteriors, and the slow dynamics separate learnable from unlearnable surprise.
- The resulting unit is the building block of FEP-ANNs. The construction derives their learning rule, explains the asymmetry between its observed and predicted terms and the temporal offset of its two factors (both previously set empirically), and shows that this offset is what makes the attractors self-orthogonalize.
- Preliminary simulations suggest that a finite jump process with only local detailed balance in its rates realizes all of this, and that networks of such units learn sequences and self-orthogonalize.
- We believe that composition into networks - rather than increasingly general derivations - is what can carry the construction toward complex adaptive systems.

(sec-one-core-several-accounts)=
### One core, several accounts

- The construction does not compete with the approaches mentioned in the Introduction; rather, it provides a physical basis for the core they share, which each of them approaches from a different angle:
  - **Bayesian brain:** posterior over hidden causes $\sim$ stationary conditional of the exposed coordinate (eq. (11)); added: a physical origin.
  - **Sampling-based representation:** samples from the posterior $\sim$ fluctuations of $\mu$ and $W$; added: variance equals susceptibility (eq. (14)).
  - **Predictive coding:** precision-weighted prediction errors $\sim$ vertical score in expectation coordinates; added: the precision of a prediction is the unit's inverse susceptibility.
  - **EM and variational inference:** alternating updates on one free energy $\sim$ fast and slow relaxation of one VFE; but the strict alternation is replaced by the timescale separation.
  - **Boltzmann machines, contrastive Hebbian learning:** driven minus free-running correlations $\sim$ horizontal score (eq. (18)); added: no free-running phase is needed, and direct fields enter only the positive phase.
  - **Equilibrium propagation, coupled learning:** clamped vs. free physical states $\sim$ driven vs. resting ensembles; added: an unsupervised, local, single-phase variant.
  - **Natural-gradient learning:** Fisher-preconditioned updates $\sim$ one admissible mobility (eq. (20)), with the same fixed points as all others.
  - **Minimum description length, learnable novelty:** learnable vs. unlearnable code length $\sim$ the two terms of the slow free energy (eq. (22)).
  - **Free energy principle:** VFE minimization $\sim$ VFE at both timescales; added: sufficient conditions, a local and conditional generative model, a blanket derived rather than assumed - a path blanket between timescales.
- Note that this is not a claim that these accounts reduce to ours: each of them addresses questions that the construction does not (neural implementation, optimality of behavior, energetic cost, design of learning materials).

(sec-relation-to-the-free)=
### Relation to the free energy principle

- **Generality.** The conditions under which a persisting system can be read as inferring appear to fail even in simple linear systems {cite:p}`10.1016/j.plrev.2021.11.001; 10.3390/e23030293`. We do not contest this. Our claim is a different one: instead of deriving inference for any persisting system, we provide sufficient conditions for a single, minimal one.
- **The generative model.** In standard formulations, internal states behind a Markov blanket model the external causes of sensory states. Here, each unit models its own exposed state given its context - the generative model is local and conditional, and a network models the whole only through the collection of its units' conditionals, akin to a pseudo-likelihood.
- **The relabeling objection** {cite:p}`10.1017/S0140525X21002351; 10.1007/s10539-021-09807-0`. Any normalized family can be described in Bayesian terms, so the inferential reading has to prove its usefulness. We believe it does so in four ways: (i) it derives the activation function and the learning rule, rather than choosing them, (ii) it gives the fixed point of learning a meaning, a criterion of success and a ceiling, (iii) it supplies a composition rule (evidence at one level acts as field at the next), and (iv) it makes contingent predictions.
- **Falsifiability.** Each assumption is an empirical question for a given substrate, and the consequences are specific:
  - the shape of the nonlinearity follows from the resting law (Langevin for uniform, $\tanh$ for binary),
  - direct input enters the observed term of the learning rule only,
  - the fluctuations of a unit should match its gain,
  - early learning should follow the correlation between context and direct input; a perfectly shielded unit should learn only from context statistics that depart from rest,
  - in networks, the order of inputs should be learned and replayed, and attractors should be approximately orthogonal - with preliminary support from human neuroimaging {cite:p}`10.7554/eLife.98725`, which however must be judged against strong null models, as high-dimensional vectors tend toward orthogonality by chance.

(sec-blankets-and-points-of)=
### Blankets and points of view

- The construction does not assume a Markov blanket, but derives one ([§ A path blanket between timescales](#sec-a-path-blanket-between)): the fast history is a path blanket of the structure - but not a state blanket, and this gap reflects what has been learned.
- At the fast level, no blanket is required: the context is not a screen, but the set of coordinates from which the structure forms its prediction. The environment may act on the exposed state directly - and must, if the unit is to learn to predict a direct input. A perfectly shielded unit ($v=0$), for which the context does screen the exposed coordinate, still infers and still learns - but only from context statistics that depart from rest, since its evidence is then the context likelihood $\log p_0(s\mid W)$ ([Supplement S4](#app-s4)).
- A *point of view* is a minimal, deflationary notion of an observer: a fast coordinate whose conditional statistics are shaped - and continually reshaped - by slow structure. Although the point of view is chosen by the modeler, it has testable consequences: the negative phase, the asymmetry of the learning rule and the precisions all follow from it. (In a regulatory network, for instance, every gene may be such a point of view on its regulators).
- In networks, the same distinction holds: a renewed state is screened by the preceding network state, which is a statement about paths, not about time slices. Blanket-like separations between groups of nodes may still emerge at the network level, but as an outcome of organization, rather than as a precondition for inference.

(sec-self-alignment-and-timescale)=
### Self-alignment and timescale stacking

- The inferential reading does not require a good model of the environment. Instead of accuracy, the geometry provides a direction: learning starts immediately, and local improvement is downhill. The geometry does not need to be designed. Its ceiling, however - set by the model family and by the physical constraints on the couplings ([§ Self-alignment](#sec-self-alignment)) - may initially be so low that alignment is hard to recognize as learning. We assume that evolution, by exploring substrates, coordinates and couplings, keeps raising this ceiling: it selects organizations whose structures can express more and more of their environment.
- We propose to order adaptive systems by the number of timescale separations they sustain:
  - *one separation:* a coarse variable tracks its forcing (a paramagnet, a chemical steady state) - an elementary inferential reading, but no learning,
  - *two separations:* a fast coordinate tracks its environment while a slower structure shapes how - the minimal unit constructed here, the first level at which perception and learning are both present,
  - *more levels:* networks of such units create a new slow timescale by themselves: their attractors are collective modes that relax more slowly than any unit ([§ Collective phenomena](#sec-collective-phenomena)), and once slow enough, they act as structure for the level below. Further levels arise in organisms and - at the slowest level - in evolution.
- **Life as a self-organized stacking of timescales.** Each separation supplies what the construction requires for the next one: coordinates that change little while faster ones relax. In this sense, timescale separation is self-catalyzing - units produce attractors, attractors provide a new slow timescale for the units above them, and evolution raises the ceiling of alignment at every level. We propose to read life as such a self-organized stacking of timescales, running from the single separation of ordinary matter through minimal adaptive units and networks to organisms and evolution.
- Each added level lengthens the horizon over which a system can integrate its past and anticipate its future - a physical reading of the expansion of a cognitive light cone {cite:p}`10.48550/arXiv.2602.08079`. This is a hypothesis rather than a result.
- **Action and planning.** The averaging step assumed presentation statistics independent of $W$; once some units feed back onto the environment, learning begins to shape the data it learns from. In FEP-ANNs, action emerges at the network level as solenoidal flow (intrinsic) or via effectors (extrinsic). Planning may emerge from timescale stacking, since one step at a slow level can span many steps below. We leave this to future work.

(sec-related-work-implications-and)=
### Related work, implications and limitations

- **Closest related work.** We share with {cite:t}`10.48550/arXiv.2602.08079` the premise that cognition is built from simpler competent parts, but we give sufficient physical conditions and derive the rules, without assuming Markov blankets at the elementary step. Learnable novelty {cite:p}`10.48550/arXiv.2607.18433` has an explicit counterpart in eq. (22); under their hierarchical prior, the description length of the couplings is largest for orthogonal directions, so self-orthogonalization can be read as storing as much independent structure as the coupling strength allows ([Supplement S7](#app-s7)). Unlike most physical learning approaches {cite:p}`10.1146/annurev-conmatphys-040821-113439; 10.1103/PhysRevX.11.021045`, no external protocol or free-running state is needed. The energetic cost of perception and learning {cite:p}`10.1103/PhysRevLett.109.120604; 10.1103/PhysRevLett.118.010601` is not computed here, and is a natural next step.
- **Dark room and noisy television.** The split of eq. (22) is the Bayesian counterpart of learnable novelty {cite:p}`10.48550/arXiv.2607.18433; 10.48550/arXiv.2601.03220`. Environments that present only resting statistics (a “dark room”) or independent noise (a “noisy television”) both leave the structure at the blind state: in the first case, because there is no surprise, and in the second, because the surprise is unpredictable and thus can not be absorbed by the structure ([Supplement S7](#app-s7)). Note, however, that a unit at the blind state is far from inactive: with $\theta=0$ and no direct field, its exposed coordinate samples the uniform resting law, i.e. it fluctuates maximally (eq. (10)). Since, in a network, the exposed coordinate is part of other units' contexts - and may also act on the environment -, a blind unit explores all possible actions on its context, rather than resting in the dark room. Exploration is thus largest exactly where the structure has learned least.
- **Implications.** The requirements are substrate-neutral, which makes the construction a candidate template for minimal models of life and protocellular cognition, for molecular programming, and for physical computing (thermodynamic, neuromorphic, memristive), where the native dynamics of the substrate implement the algorithm, and noise serves as a resource that represents uncertainty.
- **Some of the limitations:**
  - The construction provides local, sufficient conditions, not universal ones; its assumptions are strong, and whether a particular substrate satisfies them remains to be shown.
  - The mobility and the circulation are not fixed by the construction; they leave fixed points unchanged, but alter learning trajectories.
  - Hidden units (without direct evidence) do not learn in the conditional form.
  - The split into learnable and unlearnable surprise is stated for the stationary law, not along finite learning trajectories.
  - The simulations are preliminary and small, with rates chosen for clear separation; how the results degrade as the separation weakens remains to be tested.

(app-supplement)=
## Supplementary material

Equation numbers (1)–(25) refer to the main text.

(app-s1)=
### S1: The assumptions and their scope

The construction rests on seven assumptions (main text, Interim summary); the seventh is harmless for a single unit with clamped context, and becomes consequential when the context changes on the fast timescale, as in networks. Here we comment on their scope.

1. **Reference ensemble.** Positivity and regularity relative to $\rho_0$ are all that is needed for the tilt form of eq. (1); the reference may already include internal couplings, boundary conditions and earlier constraints. A globally equilibrating system is not required: the reference may itself be a non-equilibrium steady state. We take the microscopic dynamics to be a continuous-time Markov jump process on a finite state space. Coarse observables of a finite system then take finitely many values; we treat the exposed coordinate as continuous on $[-1,1]$, as the limit of many levels (already 21 levels reproduce the continuous family closely, [Figure 2B](#fig-unit)). A diffusion picture would work as well, but requires additional assumptions: (i) reflecting boundaries that keep the coordinates bounded, with a diffusion coefficient that is constant on the range, so that the resting law of $\mu$ is uniform; (ii) a definition of discrete learning events - in a diffusion every coordinate changes continuously, so renewals must be replaced, e.g. by registration at the ticks of an independent clock, or by a delay kernel in the Hebbian statistic; (iii) an explicit response latency (e.g. a delayed drift), since a diffusion responds instantaneously to its current context; and (iv) regularity conditions for the path laws (e.g. of Girsanov–Novikov type), which are automatic for finite jump processes.
2. **Double timescale separation.** A single gap suffices for an elementary form of perception (a stationary conditional under forcing is a posterior); learning needs a second gap, so that a structure changes slowly relative to the states it shapes. The presentation timescale $\tau_u$ must fall between the two, and presentation statistics must not depend on $W$. The latter excludes action at the unit level - and it is exactly the assumption that composition relaxes.
3. **Bounded coordinates.** For a coordinate constrained only by its range, the linear tilt $e^{v\mu}$ exists for every $v$ only if the range is bounded: on a half-line only one sign of $v$ is admissible, and on the whole line none is. With bounded coordinates, all linear and bilinear families are regular for all parameter values.
4. **Mesoscale sufficiency.** The environment acts on the fast coordinates and reaches the structure only through them (S2). In particular, the tilt does not alter the noise of the structural kinetics, which makes the averaged stationary law of the structure exactly $e^{\ell}$ (S3).
5. **Adiabatic conditional relaxation.** At each frozen value of the slow coordinates, the fast coordinates follow an effectively Markovian dynamics with a unique stationary conditional, and relative entropy to that conditional contracts on a timescale short compared to the motion of the structure; finite-lag and memory corrections are absorbed into effective noise. Timescale separation makes memory kernels decay rapidly, and Mori–Zwanzig projection identifies the corresponding memory and noise terms {cite:p}`10.1103/PhysRev.124.983; 10.1143/PTP.33.423`. For a Markovian reduced dynamics with an invariant conditional, relative entropy can not increase, and mixing or spectral-gap conditions make the decrease strict. The assumption is also an adequacy condition on the chosen coordinates: anything the fast sector remembers for long (e.g. another metastable basin or a conserved quantity) should be promoted to a retained coordinate. Mixing is required only within fibers; the full system need not be ergodic, in equilibrium or in detailed balance, and may carry stationary probability currents. Applied one level up, the same contraction drives the structural law toward $e^{\ell}$. Slow diffusive kinetics are therefore not a separate assumption: the diffusion of eq. (19) is the small-step limit of any local structural kinetics, such as the Glauber steps of the jump process.
6. **Leading order.** For each direction of the bundle, only the lowest-order sufficient statistic is kept; the motivation and the limits of this choice are given in S6. Each element (the bilinear interaction, the linear direct field, the uniform resting law) can be relaxed, when a substrate requires it.
7. **Causal renewal.** A unit's renewal is caused by the context at the opening of a window - $\mu^+$ is drawn from its conditional given $s$, independently of the other coordinates -, while it is registered together with the context $s^+$ at the end of the window. This fixes the timing of the two factors of the learning rule (S8, S9). Physically, the assumption requires a form of latching: the response must be set by a slightly older context than the one present when the learning event is registered. In a continuous-time jump process, in which the jump rates of $\mu$ depend on the current context, it holds exactly if the unit's field is fixed when the window opens, and approximately if the context rarely changes within a single renewal of $\mu$; otherwise, $\mu^+$ responds to a low-pass filtered context, with corrections of the order of the ratio of response latency to the timescale of context changes. We leave the mechanism open, but note that any finite response latency provides it, for instance: conduction and synaptic delays in neurons (a postsynaptic spike is caused by earlier input, while plasticity reads the presynaptic activity coincident with the spike), commitment points of molecular switches (once committed, a transition completes regardless of the field), encounter kinetics of molecular complexes (the field is set at binding, the modification is written at dissociation), delays of gene expression, or clocked and oscillation-gated systems. Synchronous renewal of all units, as in our network simulations, is the limiting case. Without latching, the rule approaches the pairing with the preceding context, which merges correlated patterns (S9); the degree of self-orthogonalization should therefore depend on the ratio of response latency to the timescale of context changes - a testable prediction.

Finally, the simple scales used to construct a unit need not resemble the emergent dynamics of networks built from many such units.

(app-s2)=
### S2: Coarse-graining the tilt

Let $z=\pi(x)$ and $W=w(z)$, so that microstates with the same $z$, and mesoscopic states with the same $W$, form nested fibers. Coarse-graining the exponential tilt gives the effective fields:

$$
u_Z(z)=\log\mathbb E_{\rho_0}[e^u\mid z],\qquad u_W(W)=\log\mathbb E_{\rho_0}[e^u\mid W],\tag{S1}
$$

related by the tower property, $e^{u_W(W)}=\mathbb E_{p_0(\cdot\mid W)}[e^{u_Z}]$. Disintegrating the tilted ensemble gives:

$$
\rho_u(dx\mid z)=e^{\,u-u_Z}\rho_0(dx\mid z),\qquad p_u(dz\mid W)=e^{\,u_Z-u_W}p_0(dz\mid W),\qquad p_u(dW)=e^{\,u_W-K(u)}p_0(dW).\tag{S2}
$$

Each factor is normalized by the effective field at the next slower level, and the overall normalizer appears only in the slowest. The main text uses the last two factors, with $Z_u(W)=e^{u_W(W)}$. The evidence also has a variational form:

$$
u_W(W)=\max_q\Big\{\mathbb E_q[u_Z]-D_{\mathrm{KL}}\big(q\,\Vert\,p_0(\cdot\mid W)\big)\Big\},\tag{S3}
$$

attained at $q=p_u(\cdot\mid W)$. This is the coarse-grained version of the minimum-relative-entropy identity:

$$
\mathbb E_\rho[u]-D_{\mathrm{KL}}(\rho\,\Vert\,\rho_0)=K(u)-D_{\mathrm{KL}}(\rho\,\Vert\,\rho_u),\tag{S4}
$$

which shows that $\rho_u$ is Jaynes's maximum-entropy ensemble relative to $\rho_0$, under the constraint $\mathbb E_\rho[u]=\mathbb E_{\rho_u}[u]$ {cite:p}`10.1103/PhysRev.106.620`. In Jaynes's usual order, the constraint is given and its multiplier found; a physical environment instead imposes the field. Learning restores Jaynes's order by dissipation: at the fixed point of learning, the environment sustains a constraint and the slow physics finds its multiplier $W^\ast$. The chain $u\to u_Z\to u_W$, in which evidence at one level acts as field at the next, is also what gives composed networks their hierarchical message-passing reading.

(app-s3)=
### S3: Averaging over presentations

Let $k(t)$ be a stationary switching process over presentations with occupation frequencies $r_k$, independent of $W$, switching on timescale $\tau_u$. While presentation $k$ is active, the fast sectors equilibrate ($\tau_z\ll\tau_u$), and the structure follows a diffusion (Itô convention):

$$
dW=\Big[\Gamma\nabla_W\big(\log Z_k(W)+\log p_0(W)\big)+\nabla_W\!\cdot\Gamma+f_{\circ,k}(W)\Big]dt+\sqrt{2\Gamma}\,dB_t.\tag{S5}
$$

Here, $\nabla_W\!\cdot\Gamma$ is the noise-induced drift required for a state-dependent mobility such as $\Gamma=\kappa g^{-1}$ to leave $e^{\log Z_k+\log p_0}$ stationary; it is of the order of the structural noise and vanishes in the low-noise limit. The tilt enters the drift only; the noise is set by the unperturbed kinetics and is common to all presentations. For $\tau_u\ll\tau_W$, the averaging principle for diffusions with rapidly switching coefficients {cite:p}`10.1007/978-0-387-73829-1` gives - to leading order in $\tau_u/\tau_W$ - an averaged diffusion with drift $\Gamma\nabla_W\ell+\nabla_W\!\cdot\Gamma+\bar f_\circ$, where $\bar f_\circ=\sum_kr_kf_{\circ,k}$, and the same noise. If the averaged circulation preserves $e^{\ell}$ (in particular, if it vanishes), the stationary law is $p(W)\propto e^{\ell(W)}$. Otherwise the circulation remains a free component that does not alter the gradient part.

**Two levels of averaging.** $\log Z_k$ is the logarithm of an average over context states under one tilt; $\ell$ averages these logarithms over presentations. The two operations do not commute - and they should not: the first is the evidence of one presentation, the second the long-run evidence of a sequence of them.

**Soft presentations.** For a general context tilt, $\log Z_k(W)=\log\int e^{u_S^{(k)}(s)+A(W\cdot s+v_k)}\,r_0(ds)-\psi(W)$, and:

$$
\nabla_W\log Z_k(W)=\mathbb E_{p_k(s\mid W)}\big[s\,L(W\cdot s+v_k)\big]-C(W),\qquad p_k(ds\mid W)\propto e^{u_S^{(k)}(s)+A(W\cdot s+v_k)}\,r_0(ds).\tag{S6}
$$

Thus, a soft presentation contributes the clamped positive phase, averaged over its own context posterior; clamping is the limit in which this posterior is a point mass.

**Sharpness and effective sample size.** The law $e^{\ell}$ concentrates when the evidence is strong or high-dimensional relative to the structural noise. How strongly accumulated evidence outweighs the prior depends on the relative rates at which evidence and prior relaxation act on $W$ in a given substrate; the construction does not fix this ratio, which appears as the weight $N$ in S7.

**Estimator noise.** Single-sample estimates of the horizontal score fluctuate on the fast timescales. After averaging, their residual contributes to $\omega_W$.

**Switches.** The averaging above treats each presentation as settled. After a switch, the fast sector is out of equilibrium for a time of order $\tau_z$, so a fraction of order $\tau_z/\tau_u$ of the renewals see a context that still reflects the previous presentation. For a single clamped unit, these renewals change nothing. In a network, with the rule of eq. (24), they add to the averaged drift a term proportional to that fraction, $\mathbb E_{\rm switch}\big[(\mu^+-L(\theta))(s^+-s)\big]$. This term is not a gradient, so the stationary law of the structure is $e^{\ell}$ only to leading order in the switch fraction. It is, however, the source of self-orthogonalization (S9).

(app-s4)=
### S4: Leaving the blind state

Integrating out $\mu$ gives $\psi(W)=\log\mathbb E_{r_0}\big[e^{A(W\cdot s)}\big]$, so for clamped presentations:

$$
\ell(W)=\mathbb E_r\big[A(W\cdot s+v)\big]-\log\mathbb E_{r_0}\big[e^{A(W\cdot s)}\big]+\log p_0(W).\tag{S7}
$$

Since $A'(0)=0$ and $A''(0)=\operatorname{Var}_{\mathrm{Unif}[-1,1]}(\mu)=1/3$, at $W=0$:

$$
\nabla_W\ell(0)=\mathbb E_r\big[s\,L(v)\big]+\nabla_W\log p_0(0),\qquad g(0)=\tfrac13\,\mathbb E_{r_0}[ss^\top].\tag{S8}
$$

With the mobility $\kappa g^{-1}$ and a prior centered at zero, the initial drift is $\dot W(0)=3\kappa\,\mathbb E_{r_0}[ss^\top]^{-1}\,\mathbb E_r\big[s\,L(v)\big]$, i.e. the regression of the direct field's effect on the context, measured in the resting metric. Whenever it is nonzero, the blind state is not a fixed point.

In the shielded case ($v=0$), the gradient vanishes at $W=0$, and the Hessian of $\ell$ there is:

$$
H=\tfrac13\left(\mathbb E_r[ss^\top]-\mathbb E_{r_0}[ss^\top]\right)+\nabla_W^2\log p_0(0).\tag{S9}
$$

For a weak prior, the linearized flow with this mobility is:

$$
\dot W\approx\kappa\,g(0)^{-1}HW\approx\kappa\left(\mathbb E_{r_0}[ss^\top]^{-1}\,\mathbb E_r[ss^\top]-I\right)W.\tag{S10}
$$

The blind state is therefore unstable along the generalized eigenvectors of the pair $\big(\mathbb E_r[ss^\top],\,\mathbb E_{r_0}[ss^\top]\big)$ with eigenvalue greater than one: the directions in which the environment excites the context more strongly than rest does. Growth is linear and Hebbian, and saturation is supplied by the nonlinearities of $L$ and $\psi$. The model is invariant under $(\mu,W)\mapsto(-\mu,-W)$, so the solutions $\pm W$ are equivalent and structural noise selects one of them; a nonzero direct field breaks this symmetry explicitly.

(app-s5)=
### S5: The fixed point as a self-consistent projection

Without the prior, $\nabla_W\ell=0$ reads $C(W^\ast)=\mathbb E_r\big[s\,L(W^\ast\!\cdot s+v)\big]$. Let:

$$
q^\ast(ds,d\mu)=\int r(ds,dv)\,p_v(d\mu\mid s,W^\ast)\tag{S11}
$$

be the environment completed by the unit's own posteriors. For fixed $q^\ast$:

$$
-D_{\mathrm{KL}}\big(q^\ast\,\Vert\,p_0(\cdot\mid W)\big)=W\cdot\mathbb E_{q^\ast}[s\mu]-\psi(W)+\text{const},\tag{S12}
$$

which is concave in $W$ and stationary exactly when $C(W)=\mathbb E_{q^\ast}[s\mu]$. The fixed point of learning is therefore the moment-matching (m-)projection of $q^\ast$ onto the unit's model family, evaluated self-consistently - which is the fixed-point condition of the EM algorithm {cite:p}`10.1016/0893-6080(95)00003-8`. What the family can not represent, remains as the residual alignment gap.

(app-s6)=
### S6: Remarks on the local normal forms

**What the leading-order form assumes.** The decomposition of eq. (7a) is exact. The simplification is to keep, for each direction of the bundle, only the lowest-order sufficient statistic ($\mu$ and $s\mu$), which gives eq. (7b). Three arguments make this restriction mild:

1. *Maximum entropy.* If the unit constrains only $\mathbb E[\mu]$ and $\mathbb E[s\mu]$, the least structured family consistent with these constraints is exactly linear and bilinear {cite:p}`10.1103/PhysRev.106.620`. The form is then not an approximation, but a stated choice of minimal sufficient statistics.
2. *Physical motivation.* If the environment couples uniformly to the microscopic degrees of freedom pooled by $\mu$ (e.g. $u(x)=h\sum_{i\in G}x_i=h|G|\,\mu$), the coarse-grained field is exactly linear in $\mu$. Likewise, uniform pair couplings between two pooled groups give exactly $W\cdot s\,\mu$. Nonlinear terms arise only from non-uniform coupling.
3. *Weak fields.* Otherwise, the linear term is the leading term of a Taylor expansion of $u_\mu$ around the operating point - accurate, however, only for weak fields, since $[-1,1]$ is not a small range.

Note that the first-order tangent argument (eq. (5a)) does not by itself justify the linear form: it guarantees the exponential (tilt) form, which eq. (1) gives exactly anyway, while the score can be any centered function. Richer statistics $T$ give the same centered score $T-\mathbb E[T]$, and thus the same construction (see below); we leave them aside, as we search for the minimal unit.

**The interaction.** The normal form $W\cdot s\,\mu$ assumes a generic operating point: the coefficient of the linear $s$–$\mu$ cross-term does not vanish identically and depends on the underlying slow variables through a map of full rank, so that it can serve as an $n$-dimensional structural coordinate. Symmetries can make this coefficient vanish; invariance under $s\mapsto-s$, for example, forbids linear coupling. The leading term is then of higher order (e.g. quadratic in $s$), and the construction goes through with the corresponding statistic in place of $s\mu$. The score structure - and with it the form of the perceptual and learning errors - is unchanged; only the sufficient statistics differ.

**The direct field.** In general, the direct field may depend on the context, $v(s)\,\mu$ with $v(s)=v+v'\cdot s+\dots$. The constant part is the leading term used in the main text. The linear part acts as an environmental contribution to the coupling: it enters the driven ensemble as an additional $s$–$\mu$ correlation, which the structure then learns to absorb into its own prediction. Higher powers of $\mu$ in the tilt change the shape of the posterior and are corrections at the operating point.

**Resting law and range.** An affine map takes any bounded range to $[-1,1]$ and rescales the natural coordinate. The uniform resting law is the maximum-entropy choice. Any other resting law $r_\mu$ on $[-1,1]$ gives the family $e^{\theta\mu-A_r(\theta)}\,r_\mu(d\mu)$, whose mean map $A_r'$ is monotone and bounded - hence sigmoid -, with limits at the ends of the support. A two-point law gives $\tanh$, and the continuous Bernoulli of FEP-ANNs is the uniform case. Only the shape of the nonlinearity depends on this choice; the tangent construction and the form of the scores do not.

(app-s7)=
### S7: Learnable and unlearnable surprise

**The slow free energy.** For a law $Q$ over structures, let:

$$
\mathfrak F_N[Q]=N\,\mathbb E_Q\big[\mathcal A(W)\big]+D_{\mathrm{KL}}\big(Q\,\Vert\,p_0\big),\tag{S13}
$$

where $N$ weighs accumulated evidence against the prior. The construction as stated has $N=1$; a substrate that accumulates evidence faster than its prior relaxes has $N>1$ (S3). Writing:

$$
\mathfrak F_N[Q]=D_{\mathrm{KL}}\big(Q\,\Vert\,Q^\ast\big)-\log\mathbb E_{p_0}\big[e^{-N\mathcal A}\big],\qquad Q^\ast\propto p_0\,e^{-N\mathcal A},\tag{S14}
$$

shows that $Q^\ast$ is the unique minimizer - equal for $N=1$ to the stationary law $e^{\ell}$ (the constant $c$ in $\mathcal A$ cancels in the normalization) -, and that the minimum is the code length of the presentations under the Bayesian mixture over the unit's model family. At the minimum, $N\,\mathbb E_{Q^\ast}[\mathcal A]$ is the residual surprise, and $D_{\mathrm{KL}}(Q^\ast\Vert p_0)$ is - by the bits-back argument - the description length of the learned structure {cite:p}`10.1145/168304.168306; 10.1109/TNN.2004.828762`.

**Laplace approximation.** With a Gaussian prior of covariance $\Sigma_0$ centered at zero, and $Q^\ast$ approximated as Gaussian around its mode $W^\ast$ with precision $NH+\Sigma_0^{-1}$ (where $H$ is the curvature of $\mathcal A$ at $W^\ast$):

$$
D_{\mathrm{KL}}\big(Q^\ast\Vert p_0\big)\approx\tfrac12\log\det\big(I+N\Sigma_0H\big)+\tfrac12\,W^{\ast\top}\Sigma_0^{-1}W^\ast-\tfrac12\operatorname{tr}\Big[I-\big(I+N\Sigma_0H\big)^{-1}\Big].\tag{S15}
$$

The last term is bounded by half the dimension of $W$, and for large $N$ the first term dominates. For isotropic $\Sigma_0=\sigma_0^2I$, each direction in which the environment constrains the structure (with curvature eigenvalue $h_i$) contributes about $\tfrac12\log(1+N\sigma_0^2h_i)$, and unconstrained directions contribute nothing. The resolution at which constrained directions are encoded is the width of $Q^\ast$, which is set by the structural noise; in the noiseless limit, the description length diverges.

**Two limiting environments.** Suppose first that the environment presents only the resting context statistics, $r(ds)=r_0(ds)$, with no direct field. Then $\ell(W)-\log p_0(W)=\mathbb E_{r_0}[A(W\cdot s)]-\log\mathbb E_{r_0}\big[e^{A(W\cdot s)}\big]\le0$ by Jensen's inequality, with equality at $W=0$: nothing is surprising, and nothing is learned. Suppose instead that the context follows its resting law and the direct fields are independent of it. The drift at $W=0$ then vanishes, since $\mathbb E_r[s\,L(v)]=\mathbb E_{r_0}[s]\,\mathbb E[L(v)]=0$ in centered coordinates, and the Hessian of the evidence there, $\big(\mathbb E[A''(v)]-\tfrac13\big)\mathbb E_{r_0}[ss^\top]$, is negative semi-definite, since $A''\le\tfrac13$. The blind state is then a stable fixed point of learning, and the surprise the fields produce is entirely unlearnable. These are the counterparts - for the minimal unit - of the dark room and the noisy television discussed by {cite:t}`10.48550/arXiv.2607.18433`.

**The log-determinant form.** For a network with coupling matrix $W$ and the hierarchical Gaussian prior with marginalized precision that {cite:t}`10.48550/arXiv.2607.18433` use to motivate their estimator, $-\log p_0(W)=\alpha\log\det\big(I+\eta WW^\top\big)+\text{const}$, so the description length of the learned couplings takes the spectral form $\alpha\sum_i\log\big(1+\eta \varsigma_i^2\big)$ in the singular values $\varsigma_i$ of $W$. For a fixed Frobenius norm and number of nonzero singular values, this sum is largest when the singular values are equal (by concavity of the logarithm): orthogonal, equally weighted coupling directions carry the most learnable structure per unit of coupling strength.

(app-s8)=
### S8: Mapping to the FEP-ANN implementation

Consider a network of nodes with states $\mu_i\in[-1,1]$. For node $i$, the exposed coordinate is $\mu_i$, the context $s=\mu_{-i}$ collects the other node states, $W$ is the $i$-th coupling row, and the direct field $v_i$ is the node's external input. The resting conditional is $p_0(\mu_i\mid\mu_{-i},W)\propto e^{(W_i\cdot\mu)\,\mu_i}$ on $[-1,1]$, and the pseudo-log-likelihood of the resting network is:

$$
\mathrm{PL}(W)=\sum_i\log p_0(\mu_i\mid\mu_{-i},W),\qquad \frac{\partial\,\mathrm{PL}}{\partial W_{ij}}=\mu_j\big(\mu_i-L(W_i\cdot\mu)\big).\tag{S16}
$$

Averaging this gradient over the driven statistics - in which each $\mu_i$ is sampled with natural parameter $W_i\cdot\mu+v_i$ - gives the rule of eq. (24) for a clamped context. The direct field appears in the sampled state, and not in the prediction. Compared to the contrastive rule, the negative phase $\mathbb E_{\rm rest}[s\mu]$ is replaced by $\mathbb E_{\rm rest}[s\mu\mid s]=s\,L(W\cdot s)$, which requires no free-running phase. Maximizers of the expected pseudo-likelihood recover the couplings when the driven statistics belong to the model family {cite:p}`10.2307/2987782`. When the context renews within the same window as the node, however, the pseudo-likelihood gradient pairs the error with the context that formed the prediction, while eq. (24) pairs it with the renewed context. Only the latter estimates the joint score, and only the latter orthogonalizes the learned attractors (S9).

(app-s9)=
### S9: Temporal contrast to leading order

Take network states $x=a\xi$, with $\xi$ the presented pattern, and the linear mean map $L(h)\approx h/3$. Write $\bar m$ for the mean pattern, $\Sigma=\mathbb E[\xi\xi^\top]$ and $\Sigma_c=\Sigma-\bar m\bar m^\top$. The expected updates of the rule $s^+(\mu^+-L(W\cdot s))$ and of its two controls are:

| Renewal | Network rule (eq. (24)) | Preceding-context pairing | Reset state |
| --- | --- | --- | --- |
| Settled | $a^2(\Sigma-W\Sigma/3)$ | same | same |
| After a switch ($l\to k$, independent) | $a^2(\Sigma-W\bar m\bar m^\top/3)$ | $a^2(\bar m\bar m^\top-W\Sigma/3)$ | $a^2\Sigma$ |

Weighting the two kinds of renewal by $1-f$ and $f$, and solving on the span of the patterns gives:

| Pairing | Fixed point $W^\ast$ | Gain of the shared relative to the distinguishing directions |
| --- | --- | --- |
| Network rule | $3\Sigma[(1-f)\Sigma_c+\bar m\bar m^\top]^{-1}$ | $1-f$ |
| Preceding context | $3(I-f\Sigma_c\Sigma^{-1})$ | $1/(1-f)$ |
| Reset | $3P/(1-f)$ | $1$ |

The gains are stated for a mean orthogonal to the centered span. Since $\Sigma_c\bar m=0$ in this case, the gain along the shared direction $\bar m$ is 3 for the first two pairings; only the gains of the distinguishing directions differ. In encounter kinetics, $f$ is the probability that a context coordinate has not renewed since the switch, $f=1/(1+\kappa\tau_u)$. For sequential presentations, with $l$ the fixed predecessor of $k$, the switch term of the network rule becomes $a^2(\Sigma-W\Sigma_{\rm lag}/3)$ with $\Sigma_{\rm lag}=\mathbb E[\xi_l\xi_k^\top]$, which makes $W^\ast$ asymmetric: sequence learning, by the same mechanism.
