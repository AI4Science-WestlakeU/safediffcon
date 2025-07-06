cd safediffcon

conda create --yes --name safediffcon python=3.8.18
conda activate safediffcon

pip install multiprocess==0.70.15
pip install matplotlib==3.2.0
pip install imageio==2.34.1
pip install scipy==1.8.0

pip install torchvision
pip install tqdm
pip install IPython

pip install -r requirements.txt
conda install numpy==1.19.0 -y