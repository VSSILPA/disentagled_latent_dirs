# Self-supervised Enhancement of Latent Discovery in GANs
[Paper](link)

Self-supervised Enhancement of Latent Discovery in GANs. \
[Silpa V S](silpavs.43@gmail.com)*,[Adarsh K](kadarsh22@gmail.com)*, [S Sumitra](https://www.iist.ac.in/mathematics/sumitra)
* indicates equal contribution.
*AAAI 2022*

## Prerequisites
- Ubuntu
- Python 3
- NVIDIA GPU + CUDA CuDNN

## Abstract
Several methods for discovering interpretable directions in the latent space of pretrained GANs have been proposed. Latent semantics discovered by unsupervised methods are relatively less disentangled than supervised methods since they do not
use pre-trained attribute classifiers. We propose Scale Ranking Estimator (SRE),which is trained using self-supervision. SRE enhances the disentanglement in directions obtained by existing unsupervised disentanglement techniques. These directions are updated to preserve the ordering of variation within each direction in latent space.


<a name="setup"/>
<a name="application"/>

## Setup

- Clone this repo:
```bash
git clone https://github.com/kadarsh22/disentanglement_based_active_learning.git
cd disentanglement_based_active_learning
```

- Install dependencies:
	- Install dependcies to a new virtual environment.
	```bash
	pip install -r requirements.txt
	```
 
## Disovered directions
<img src='discovered_directions.png' width=800>

## Application
<img src='image_retrival.png' width=800>

