
To train/eval you can use the following scripts:


 * [Training script](train.sh) (you might need to chmod +x the file)
   * We have the following options:
     * ```-d [String]``` : Path to the dataset
     * ```-a [String]```: Path to the Architecture configuration file 
     * ```-l [String]```: Path to the main log folder
     * ```-n [String]```: additional name for the experiment
     * ```-c [String]```: GPUs to use (default ```no gpu```)
     * ```-u [String]```: If you want to train an Uncertainty version of 3Dnet (default ```false```) 
   * For example if you have the dataset at ``/dataset`` the architecture config file in ``/3Dnet.yml``
   and you want to save your logs to ```/logs``` to train "3Dnet" with 2 GPUs with id 3 and 4:
     * ```./train.sh -d /dataset -a /3Dnet.yml -m 3Dnet -l /logs -c 3,4```
<br>
<br>

 * [Eval script](eval.sh) (you might need to chmod +x the file)
   * We have the following options:
     * ```-d [String]```: Path to the dataset
     * ```-p [String]```: Path to save label predictions
     * ``-m [String]``: Path to the location of saved model
     * ``-s [String]``: Eval on Validation or Train (standard eval on both separately)
     * ```-u [String]```: If you want to infer using an Uncertainty model (default ```false```)
     * ```-c [Int]```: Number of MC sampling to do (default ```30```)
   * If you want to infer&evaluate a model that you saved to ````/3Dnet/logs/[the desired run]```` and you
   want to infer$eval only the validation and save the label prediction to ```/pred```:
     * ```./eval.sh -d /dataset -p /pred -m /3Dnet/logs/[the desired run] -s validation -n 3Dnet```
     

