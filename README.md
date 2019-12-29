# drone
Execução de comandos para preparar o ambiente

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install git

# Processo de instalação das ferramentas
1. Instalar Ubuntu 14.04 (pode usar outra versão, contudo tem que adaptar as versões para cada biblioteca como por exemplo o tensorflow)

2. Instalar python 2.7

```bash
sudo apt-get update
sudo apt-get install build-essential checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev

sudo apt-get install python-virtualenv
virtualenv --system-site-packages -p python2.7 ~/drone_project
source ~/drone/bin/activate
```
3. Instalar keras

```bash
sudo apt-get install python-pip
sudo apt-get install python-dev -—upgrade pip
sudo -H pip install numpy
sudo -H pip install scipy
sudo -H pip install matplotlib 
sudo -H pip install scikit-image
sudo -H pip install scikit-learn
sudo -H pip install ipython 
sudo -H pip install protobuf
sudo -H pip install pyyaml
sudo -H pip install HDF5
sudo -H pip install h5py
sudo -H pip install --upgrade keras
```

4. Instalar tensorflow

```bash
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0rc2-cp27-none-linux_x86_64.whl
```
5. Instalar OpenCV 3
```bash
pip install opencv-python==3.3.0.10
```

6. Verifique se tudo foi instalado corretamente

```bash
source ~/drone/bin/activate
python
import numpy
numpy.__version__
import tensorflow
tensorflow.__version__
import keras
keras.__version__
import opencv
opencv.__version__
```

7. Instalar ROS

```link
http://wiki.ros.org/indigo/Installation/Ubuntu
```

8. Instalar Gazebo
```link
http://gazebosim.org/tutorials?tut=install_ubuntu&cat=install
```
9. Instalar e criar catkin workspace

```link
http://wiki.ros.org/catkin/Tutorials/create_a_workspace
```

10. Clonar Projeto

ir dentro do projeto catkin criado e entrar na pasta ```src``` e executar:
```bash
https://github.com/davidbrahiam/drone.git
```

11. Ativar e instalar pre-requisitos do projeto

Ir para o projeto catkin criado e executar:

```bash
source devel/setup.bash
```

# Treinamento da rede neural
Acesse a pasta tcc_train.
Para Treinar com o MLP
```bash
python train_simple_nn.py --dataset images --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png --roc mlp_curve.png
```

Para Treinar com o CNN
```bash
python train_vgg1.py --dataset images --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --plot output/smallvggnet_plot.png --roc cnn_curve.png
```

Os graficos e resultados serão no local passado como parametro. Neste caso na pas output será apresentando as saidas dos resultados.

O codigo do cnn é divido entre o arquivo ```train_vgg1.py```(Execução do treinamento) e ```pyimagesearch/lenet.py```(definição do modelo CNN).

OBS: Devido que o tensorflow usado neste trabalho não suporta a função de ativiação softmax, este foi substituido por uma sigmoid e teve sua cross_entropy alterada para binary_entropy.

Para realizar os testes visualizar arquivo ```predict.py``` e para verificar a navegação do drone, acessar ```tcc_navegate.py```

Qualquer dúvida abrir issue.