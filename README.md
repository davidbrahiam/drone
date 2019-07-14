# drone
Execução de comandos para preparar o ambiente

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install git

# Treinamento da rede neural
Acesse a pasta tcc_train.
Para Rodar com o Perceptron
```bash
python train_simple_nn.py --dataset images --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png
```

Para Rodar com o CNN
```bash
python train_vgg.py --dataset images --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --plot output/smallvggnet_plot.png
```

Os graficos e resultados serão no local passado como parametro. Neste caso na pas output será apresentando as saidas dos resultados.

O codigo do cnn é divido entre o arquivo ```train_vgg.py```(Execução do treinamento) e ```pyimagesearch/smallggnet.py```(definição do modelo CNN).

OBS: Devido que o tensorflow usado neste trabalho não suporta a função de ativiação softmax, este foi substituido por uma sigmoid e teve sua cross_entropy alterada para binary_entropy.

# A parte da explicação de instalação e simulação do ambiente está pendente
