# A GAN algorithm to produce weak lensing convergence maps and cosmic web slices: 


This repository contains a GAN algorithm that allows producing novel cosmic web slices (overdensity field in 2-D) and weak lensing convergence maps of different redshifts, cosmological parameters and modifications of gravity. In addition, the code allows producing novel, statistically realistic cosmic web slices using the publicly available Illustris data.   

The figures below illustrate some of the results produced by our code.

<p align="center">
<img src="./images/cosmoGAN_cw_wl_samples.png?raw=true" width="700">
</p>

The figure below shows a few GAN-generated cosmic web slices produced using Illustris data. Namely, it shows the dark matter overdensity, gas overdensity and internal (thermal) energy fields correspondingly along with all the three mentioned components combined into an RGB array:  

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

If you require GPU support, install the GPU version instead: **pip install tensorflow-gpu==1.14.0** instead.

**Cloning the repository and getting the data:**

Clone the repository:

```
git clone https://github.com/AndriusT/cw_wl_GAN.git
```

This contains all the key code and the required Jupyter notebooks, however, you still need to download the [*training data*](https://drive.google.com/open?id=10h827ENwIfqjQg3yIXF5W-5oYbKKb_Bp) and the saved [*checkpoint files*](https://drive.google.com/open?id=1LxQZODfKyDfJM1bTanHhVY98dXVpPbiw). Download the files and move then to the ./data and the ./checkpoints folders correspondingly. The files can be extracted using p7zip-full (**sudo apt-get install p7zip-full**): 

```
cd ./data
7z e data.7z
cd ../checkpoints/
7z x checkpoints.7z
```

The checkpoints include the most recent results for weak lensing maps with different cosmological parameters along with the cosmic web slices of different redshifts, cosmologies and modified gravity parameters. The data folder includes samples of the training data used to trained our algorithm. 

## Getting the Illustris data:

To get the Illustris snapshot data (needed in the [*Preparing_Illustris_data.ipynb*](./Preparing_Illustris_data.ipynb) notebook), use the following code:

```
cd ./data/snapdir_135/
wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 --content-disposition --header="API-Key: your-api-code-here" "http://www.illustris-project.org/api/Illustris-3/files/snapshot-135/?format=api"
```

Your personal api code has to be added to make this work. To get the code, register at the [*Illustris website*](https://www.illustris-project.org/). This will download a z = 0.0 snapshot file (21.8 GB), which can then be used to extract the dark matter and gas overdensity slices. 

## Citing the code:

If you use these codes in your research, please cite this repository and the corresponding paper

Also, please cite the [*cosmoGAN paper*](https://arxiv.org/abs/1706.02390), which this work is based on. Also, see the [*github repository*](https://github.com/MustafaMustafa/cosmoGAN). 

## Contacting the author: 

[*Contact me*](mailto:andrius.tamosiunas@port.ac.uk) in case of any issues with the code. 
