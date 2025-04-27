# csc311-homework-3-solved
**TO GET THIS SOLUTION VISIT:** [CSC311 Homework 3 Solved](https://www.ankitcodinghub.com/product/csc311-homework-3-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;91520&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSC311 Homework 3 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column"></div>
</div>
<div class="layoutArea">
<div class="column">
&nbsp;

In this question, you will derive the backprop updates for a particular neural net architecture. The network is similar to the multilayer perceptron architecture from lecture, and has one linear hidden layer. However, there are two architectural differences:

<ul>
<li style="list-style-type: none;">
<ul>
<li>In addition to the usual vector-valued input x, there is a vector-valued “context” input η. (The particular meaning of η isn’t important for your derivation, but think of it as containing additional task information, such as whether to focus on the left or the right half of the image.) The hidden layer activations are modulated based on η; this means they are multiplied by a value which depends on η.</li>
<li>The network has a skip connection which sends information directly from the input to the output of the network.The loss function is squared error. The forward pass equations and network architecture are as follows (the symbol ⊙ represents elementwise multiplication, and σ denotes the logistic function):
z = Wx s = Uη

h = z ⊙ σ(s)

y = v⊤h + r⊤x

L = 12 ( y − t ) 2

The model parameters are matrices W and U, and vectors v and r. Note that there is only

one output unit, i.e. y and t are scalars.

1 http://www.cs.toronto.edu/~rgrosse/courses/csc311_f21/syllabus.pdf
</li>
</ul>
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
r

</div>
<div class="column">
y

v

U

</div>
</div>
<div class="layoutArea">
<div class="column">
h

</div>
</div>
<div class="layoutArea">
<div class="column">
x

</div>
</div>
<div class="layoutArea">
<div class="column">
W

</div>
</div>
<div class="layoutArea">
<div class="column">
⌘

</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
(a) [1pt] Draw the computation graph relating x, z, η, s, h, and the model parameters.

(b) [4pts] Derive the backprop formulas to compute the error signals for all of the model parameters, as well as x and η. Also include the backprop formulas for all intermediate quantities needed as part of the computation. You may leave the derivative of the logistic function as σ′ rather than expanding it out explicitly.

2. [13pts] Fitting a Na ̈ıve Bayes Model

In this question, we’ll fit a Na ̈ıve Bayes model to the MNIST digits using maximum likeli- hood. In addition to the mathematical derivations, you will complete the implementation in naive_bayes.py. The starter code will download the dataset and parse it for you: Each training sample (t(i),x(i)) is composed of a vectorized binary image x(i) ∈ {0,1}784, and 1-of-10 encoded class label t(i), i.e., t(i) = 1 means image i belongs to class c.

Given parameters π and θ, Na ̈ıve Bayes defines the joint probability of the each data point x and its class label c as follows:

784

p(x,c|θ,π) = p(c|θ,π)p(x|c,θ,π) = p(c|π)􏰄p(xj |c,θjc). j=1

where p(c|π) = πc and p(xj = 1|c,θ,π) = θjc. Here, θ is a matrix of probabilities for each pixel and each class, so its dimensions are 784 × 10, and π is a vector with one entry for each class. (Note that in the lecture, we simplified notation and didn’t write the probabilities conditioned on the parameters, i.e. p(c|π) is written as p(c) in lecture slides).

For binary data (xj ∈ {0, 1}), we can write the Bernoulli likelihood as

p(xj | c, θjc) = θxj (1 − θjc)(1−xj ), (1)

which is just a way of expressing p(xj = 1|c, θjc) = θjc and p(xj = 0|c, θjc) = 1 − θjc in a compact form. For the prior p(t|π), we use a categorical distribution (generalization of Bernoulli distribution to multi-class case),

9

p(tc = 1|π) = p(c|π) = πc or equivalently p(t|π) = Π9j=0πtj where 􏰃πi = 1,

i=0

</div>
</div>
<div class="layoutArea">
<div class="column">
c

</div>
</div>
<div class="layoutArea">
<div class="column">
jc

</div>
</div>
<div class="layoutArea">
<div class="column">
where p(c | π) and p(t | π) can be used interchangeably. You will fit the parameters θ and π using MLE and MAP techniques. In both cases, your fitting procedure can be written as a few simple matrix multiplication operations.

</div>
</div>
<div class="layoutArea">
<div class="column">
(a) [3pts] First, derive the maximum likelihood estimator (MLE) for the class-conditional pixel probabilities θ and the prior π. Derivations should be rigorous.

Hint 1: We saw in lecture that MLE can be thought of as ‘ratio of counts’ for the data, so what should θˆ be counting?

</div>
</div>
<div class="layoutArea">
<div class="column">
jc

Hint 2: Similar to the binary case, when calculating the MLE for πj for j = 0, 1, …, 8,

</div>
</div>
<div class="layoutArea">
<div class="column">
t(i)

write p(t(i) | π) = Π9 π j and in the log-likelihood replace π9 = 1 − Σ8 πj , and then

</div>
</div>
<div class="layoutArea">
<div class="column">
j=0 j j=0

take derivatives w.r.t. πj . This will give you the ratio πˆj /πˆ9 for j = 0, 1, .., 8. You know

</div>
</div>
<div class="layoutArea">
<div class="column">
that πˆj’s sum up to 1.

(b) [1pt] Derive the log-likelihood log p(t|x, θ, π) for a single training image.

</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
<div class="layoutArea">
<div class="column">
j

(c) [3pt] Fit the parameters θ and π using the training set with MLE, and try to report the average log-likelihood per data point N1 ΣNi=1 log p(t(i)|x(i), θˆ, πˆ), using Equation (1). What goes wrong? (it’s okay if you can’t compute the average log-likelihood here).

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
<ol start="4">
<li>(d) &nbsp;[1pt] Plot the MLE estimator θˆ as 10 separate greyscale images, one for each class.</li>
<li>(e) &nbsp;[2pt] Derive the Maximum A posteriori Probability (MAP) estimator for the class- conditional pixel probabilities θ, using a Beta(3, 3) prior on each θjc. Hint: it has a simple final form, and you can ignore the Beta normalizing constant.</li>
<li>(f) &nbsp;[2pt] Fit the parameters θ and π using the training set with MAP estimators from previ-

ous part, and report both the average log-likelihood per data point, N1 ΣNi=1 log p(t(i)|x(i), θˆ, πˆ), and the accuracy on both the training and test set. The accuracy is defined as the frac-

tion of examples where the true class is correctly predicted using cˆ = argmaxc log p(tc =

1|x, θˆ, πˆ ).</li>
<li>(g) &nbsp;[1pt] Plot the MAP estimator θˆ as 10 separate greyscale images, one for each class.</li>
</ol>
3. [7pts] Categorial Distribution. In this problem you will consider a Bayesian approach to modelling categorical outcomes. Let’s consider fitting the categorical distribution, which is a discrete distribution over K outcomes, which we’ll number 1 through K. The probability of each category is explicitly represented with parameter θk. For it to be a valid probability distribution, we clearly need θk ≥ 0 and 􏰂k θk = 1. We’ll represent each observation x as a 1-of-K encoding, i.e, a vector where one of the entries is 1 and the rest are 0. Under this model, the probability of an observation can be written in the following form:

K

p(x|θ) = 􏰄 θxk . k

k=1

Suppose you observe a dataset,

Denote the count for outcome k as Nk = 􏰂n x(i). Recall that each data point is in the

</div>
</div>
<div class="layoutArea">
<div class="column">
D = {x(i)}Ni=1. i=1 k

</div>
</div>
<div class="layoutArea">
<div class="column">
1-of-K encoding, i.e., x(i) = 1 if the ith datapoint represents an outcome k and x(i) = 0 kk

otherwise. In the previous assignment, you showed that the maximum likelihood estimate for

</div>
</div>
<div class="layoutArea">
<div class="column">
the counts was:

</div>
<div class="column">
θˆ k = N k . N

</div>
</div>
<div class="layoutArea">
<div class="column">
<ol>
<li>(a) &nbsp;[2pts] For the prior, we’ll use the Dirichlet distribution, which is defined over the set of probability vectors (i.e. vectors that are nonnegative and whose entries sum to 1). Its PDF is as follows:p(θ) ∝ θa1−1 ···θak−1. 1K
Determine the posterior distribution p(θ|D). Based on your answer, is the Dirichlet distribution a conjugate prior for the categorial distribution?
</li>
<li>(b) &nbsp;[3pts] Still assuming the Dirichlet prior distribution, determine the MAP estimate of the parameter vector θ. For this question, you may assume each ak &gt; 1.

Hint: Remember that you need to enforce the constraint that 􏰂k θk = 1. You can do this using the same parameterization trick you used in Question 2. Alternatively, you could use Lagrange multipliers, if you’re familiar with those.</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column"></div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
&nbsp;

</div>
</div>
<div class="layoutArea">
<div class="column">
(c) [2pts] Now, suppose that your friend said that they had a hidden N + 1st outcome, x(N+1), drawn from the same distribution as the previous N outcomes. Your friend does not want to reveal the value of x(N+1) to you. So, you want to use your Bayesian model to predict what you think x(N+1) is likely to be. The “proper” Bayesian predictor is the so-called posterior predictive distribution:

􏰅

p(x(N+1)|D) =

What is the probability that the N +1 outcome was k, i.e., the probability that x(N+1) =

</div>
</div>
<div class="layoutArea">
<div class="column">
p(x(N+1)|θ)p(θ|D) dθ

1, under your posterior predictive distribution? Hint: A useful fact is that if θ ∼

</div>
</div>
<div class="layoutArea">
<div class="column">
k

4. [5pts] Gaussian Discriminant Analysis. For this question you will build classifiers to label images of handwritten digits. Each image is 8 by 8 pixels and is represented as a vector of dimension 64 by listing all the pixel values in raster scan order. The images are grayscale and the pixel values are between 0 and 1. The labels y are 0, 1, 2, . . . , 9 corresponding to which character was written in the image. There are 700 training cases and 400 test cases for each digit; they can be found in the data directory in the starter code.

A skeleton (q4.py) is is provided for each question that you should use to structure your code. Starter code to help you load the data is provided (data.py). Note: the get digits by label function in data.py returns the subset of digits that belong to a given class.

Using maximum likelihood, fit a set of 10 class-conditional Gaussians with a separate, full covariance matrix for each class. Remember that the conditional multivariate Gaussian prob- ability density is given by,

</div>
</div>
<div class="layoutArea">
<div class="column">
Dirichlet(a1,…,aK), then

</div>
<div class="column">
ak E[θk] = 􏰂 .

</div>
</div>
<div class="layoutArea">
<div class="column">
Report your answers to the above questions.

</div>
</div>
<div class="layoutArea">
<div class="column">
−d/2 −1/2 􏰀 1 T −1 􏰁 p(x|y=k,μ,Σk)=(2π) |Σk| exp −2(x−μk) Σk (x−μk)

</div>
<div class="column">
(2)

</div>
</div>
<div class="layoutArea">
<div class="column">
You should take p(y = k) = 1 . You will compute parameters μkj and Σk for k ∈ (0…9), j ∈ 10

(1…64). You should implement the covariance computation yourself (i.e. without the aid of np.cov). Hint: To ensure numerical stability you may have to add a small multiple of the identity to each covariance matrix. For this assignment you should add 0.01I to each matrix.

<ol>
<li>(a) &nbsp;[3pts] Using the parameters you fit on the training set and Bayes rule, compute the average conditional log-likelihood, i.e. N1 􏰂Ni=1 log(p(y(i) | x(i), θ)) on both the train and test set and report it. Hint: for numerical stability, you will want to use np.logaddexp, as discussed in Lecture 4.</li>
<li>(b) &nbsp;[1pt] Select the most likely posterior class for each training and test data point as your prediction, and report your accuracy on the train and test set.</li>
<li>(c) &nbsp;[1pt] Compute the leading eigenvectors (largest eigenvalue) for each class covariance matrix (can use np.linalg.eig) and plot them side by side as 8 by 8 images.</li>
</ol>
Report your answers to the above questions, and submit your completed Python code for q4.py.

</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
<div class="layoutArea">
<div class="column">
k′ ak′

</div>
</div>
</div>
