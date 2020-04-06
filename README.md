# A GAN algorithm to produce weak lensing convergence maps and cosmic web slices: 

This is a code that reproduces some of the key results from the paper: **arxiv link here**. 

In particular it contains a variation of the original [*cosmoGAN*](https://github.com/MustafaMustafa/cosmoGAN) code. It allows producing novel cosmic web slices (overdensity field in 2-D) and weak lensing convergence maps of different redshifts, cosmological parameters and modifications of gravity. In addition, the code allows producing novel, statistically realistic cosmic web slices using the publicly available Illustris data.   

The figures below illustrate some of the results produced by our code.

<p align="center">
<img src="./images/cosmoGAN_cw_wl_samples.png?raw=true" width="700">
</p>

The figure below shows a few GAN-produced cosmic web slices produced using Illustris data. Namely, it shows the dark matter overdensity, gas overdensity and internal (thermal) energy fields correspondingly along with all the three mentioned components combined into an RGB array:  

<p align="center">
<img src="./images/illustris_GAN_samples.png?raw=true" width="700">
</p>

## How to run:

**Installing the dependencies:**

The key python packages used in the attached Jupyter notebooks are the following: tensorflow (1.14.0), cv2 (4.1.2), numpy (1.16.2), matplotlib (3.0.3) and nbodykit (0.3.12). The easiest way to make sure you have the same versions of the mentioned packages is to create a fresh Conda environment as follows:

```
conda create -n myenv python=3.6
source activate myenv
pip install tensorflow==1.14.0
pip install opencv-python
pip install matplotlib
pip install numpy
conda install -c bccp nbodykit
```

Note that nbodykit also requires *cython* and *mpi4py* to be installed. For more information see [*here*](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html).

**Cloning the repository and getting the data:**

Clone the repository by typing the following:

```
git clone https://github.com/AndriusT/cw_wl_GAN.git
```

This contains all the key code and the required Jupyter notebooks, however, you still need to download the training data and the saved checkpoint files. To do this, navigate to the data folder and download and extract the data file: 

```
cd ./data
wget https://drive.google.com/open?id=10h827ENwIfqjQg3yIXF5W-5oYbKKb_Bp
7z e data.7z
```

To download the checkpoint files:

```
cd ../checkpoints/
wget https://drive.google.com/open?id=1LxQZODfKyDfJM1bTanHhVY98dXVpPbiw
7z e checkpoints.7z
```

Where we used p7zip-full (sudo apt-get install p7zip-full) to extract the files.
