<!-- $size: 16:9 -->
<link rel='stylesheet' href='../assets/slides.marp.css'>

# Neural Language Processing with PyTorch

**Week 1** 딥러닝을 위한 PyTorch 실무환경 구축

---

### Ki Hyun Kim

<div style="float: right; width: 20rem;"><img src="../assets/kihyun.png"></div>

* Machine Learning Researcher @ MakinaRocks
* Linkedin: https://www.linkedin.com/in/ki-hyun-kim/
* Github: https://github.com/kh-kim/
* Email: pointzz.ki@gmail.com

---

### Ki Hyun Kim

<div style="position: absolute; right: 0; bottom: 0; width: 25rem;"><img src="../assets/11st.png"></div>

* Machine Learning Researcher @ SKPlanet
  * Neural Machine Translation
  * 글로벌 11번가
    * 한영/영한, 한중/중한 기계번역
    * 7000만 개 이상의 상품타이틀 번역, 리뷰 실시간 번역
  * SK AI asset 공유
    * SK C&C Aibril: 한중/중한, 한영/영한, 영중/중영 API 제공
    * SK 그룹 한영중 통번역기 API 제공

---

### Ki Hyun Kim

<div style="position: absolute; right: 5rem; bottom: -9rem; width: 22rem;"><img src="../assets/genie.png"></div>

* Machine Learning Engineer @ TMON
  * Recommender System
  * QA-bot
* Researcher @ ETRI
  * Automatic Speech Translation
  * GenieTalk
* BS + MS of CS @ Stony Brook Univ.

---

### 오상준

<div style="float: right; width: 20rem;"><img src="../assets/june.jpg"></div>

- Deep Learning Engineer @ Deep Bio
  - 병리영상 기반 전립선암 진단모델 연구개발
  - GPU 서버 분산 스케쥴링 시스템 개발
- Co-founder, Research Engineer @ QuantumSurf
  - 선물거래 알고리즘을 위한 API 설계 및 UX 개발
  - IPTV 영상품질 예측모델 연구개발
- BS in English Literature, minor in Philosophy @ 한국외국어대학교

---

### 오상준

* Github: https://github.com/juneoh
* Email: me@juneoh.net

---

<!-- *template: section -->

## 1. Introduction to Deep Learning

---

### Genealogy

![bg](../assets/diagram.png)

---

### Timeline

* **Cybernetics** 1940s-1960s
  * McCulloh-Pitts neuron
    * McCulloch and Pitts, 1942.  A Logical Calculus of the Ideas Immanent in Nervous Activity.
  * Hebbian learning
    * Hebb, 1949.  The Organization of Behaviour.
  * Perceptron
    * Rosenblatt, 1958. The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.

<!-- *footer: https://www.deeplearningbook.org/ -->

---

<!-- *template: center -->

![](../assets/mlp.jpg)
![](../assets/winter.jpg)

---

### Timeline

* **Connectionism** 1980s-1990s
  * Backpropagation
    * Rumelhart et al, 1986. Learning Representations by Back-propagating Errors.
  * Convolutional Neural Networks
    * Fukushima, 1980. Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position.

<!-- *footer: https://www.deeplearningbook.org/ -->

---

### Timeline

* **Deep Learning** 2006-
  * Deep Neural Networks
    * Hinton et al, 2006. A Fast Learning Algorithm for Deep Belief Nets.
  * Rectified Linear Units
    * Golorot et al, 2011. Deep Sparse Rectifier Neural Networks.
  * AlexNet
    * Krizhevsky et al, 2012. ImageNet Classifification with Deep Convolutional Neural Networks.

<!-- *footer: https://www.deeplearningbook.org/ -->

---

### Neural Networks

* Feed-forward Network `y = network(x)`

<center><img src='../assets/neuron.jpg' /></center>

---

### Neural Networks

* Feed-forward Network `y = network(x)`

<center><img style='height:22rem' src='../assets/artificial_neuron.png' /></center>

---

### Neural Networks

* Feed-forward Network `y = network(x)`

<center><img style='height:28rem' src='../assets/fnn.png' /></center>

---

### Neural Networks

* Backpropagation `(loss_function(y_pred, y_true)).backward()`

<center><img style='height:25rem' src='../assets/backprop.png' /></center>

<!-- *footer: http://colah.github.io/posts/2015-08-Backprop/ -->

---

### Neural Networks

* Gradient Descent [`torch/optim`](https://pytorch.org/docs/stable/optim.html)
  * Stochastic Gradient Descent, Momentum, Adagrad, Adam, ...

<center><img style='height:25rem' src='../assets/gradient_descent.png' /></center>

---

<!-- *template: center -->

<center><img style='height:25rem' src='../assets/optimization.gif' /></center>

---

### Activation functions and non-linearity

* $sigmoid(x) = \frac{1}{1 + e^x}$ [`torch.nn.Sigmoid`](https://pytorch.org/docs/stable/nn.html#torch.nn.Sigmoid)<br>

<center><img style='height:25rem' src='../assets/sigmoid.png' /></center>

<!-- *footer: https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions -->

---

### Activation functions and non-linearity

* $relu(x) = max(0, x)$ [`torch.nn.ReLU`](https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU)<br>

<center><img style='height:25rem' src='../assets/relu.png' /></center>

---

### Activation functions and non-linearity

* $softmax(z) = \frac{e^{z_{j}}}{\sum^K_{k=1}{e^{z_k}}}$ [`torch.nn.Softmax`](https://pytorch.org/docs/stable/nn.html#torch.nn.Softmax)<br>
<br/>
<center><img style='height:15rem' src='../assets/softmax.png' /></center>

---

### Loss functions

* L1 loss<br>
  $L_1 = \sum^n_{i=1}|y_i - \hat{y}_i|$
  <br>
  
* L2 loss<br>
  $L_2 = \sum^n_{i=1}(y_i - \hat{y}_i)^2$

---

### Loss functions


* Mean Square Error [`torch.nn.MSELoss`](https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss)<br>
  $MSE = \frac{1}{n}\sum^n_{i=1}(y_i - \hat{y_i})^2$
<br>
* Cross Entropy [`torch.nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss)<br>
  $CE = -\sum^n_{i=1}y_iln(\hat{y_i})$

---

### Loss functions

* Binary Cross Entropy [`torch.nn.BCELoss`](https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss)<br>
  $BCE = -\frac{1}{n}\sum^n_{i=1}[y_iln(\hat{y_i}) + (1 -y_i)ln(1 - \hat{y_i})]$

---

### Regularization methods

<center><img src='../assets/polynomial.jpg'></center>

---

### Regularization methods

* Weight decay<br>
  $W \leftarrow W - \lambda(\frac{\partial{L}}{\partial{W}} + \gamma\lVert W \rVert)$

---

### Regularization methods

* Dropout
<center><img style='height:25rem' src='../assets/dropout.png' /></center>

<!-- *footer: Bishop. <Pattern Recognition and Machine Learning.> -->

---

<!-- *template: section -->

## 2. Hello PyTorch

---

<br>
<center><img style='height:5rem' src='../assets/pytorch.png' /></center>
<br>

* Deep Learning Framework
  * Tensorflow, Keras, Torch, Chainer, MXNet
* Python-native, NumPy-friendly
* Dynamic graphs
* https://pytorch.org/
* https://pytorch.org/docs/stable/index.html

---

![bg](../assets/dynamic_graph.png)

---

### Our stack

* **Ubuntu 16.04**
  * Basic shell functions: 
    `ls -lah`, `cd`, `cp`, `mv`, `cat`, `grep -irv`, `kill`, `ps -ef`, `history`, `tar`, `unzip`, `apt`, `curl`
<br>
<center><img style='height:14rem' src='../assets/ubuntu.jpg' /></center>

---

### Our stack

* **Python 3.6+**

```
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
```

---

### Our stack

* **Conda**
  * Package manager + virtual environments
  * https://conda.io/
<br>
<center><img style='height:5rem' src='../assets/conda.svg' /></center>

---

### Our stack

* **Jupyter Notebook**
  * Document and visualize live code
  * http://jupyter.org/
<br>
<center><img style='height:7rem' src='../assets/jupyter.png' /></center>

---

### Our stack

* **git**
  * Version control system
  * Basic commands:
    `git clone`, `git checkout`, `git add`, `git commit`, `git push`, `git pull`, `git diff`, `git log`
  * https://github.com/, https://gitlab.com/

---

### Our stack

* Docker CE
  * Container virtualization
  * Basic commands:
    `docker run`, `docker exec`, `docker start`, `docker stop`

<center><img style='height:14rem' src='../assets/docker.png' /></center>