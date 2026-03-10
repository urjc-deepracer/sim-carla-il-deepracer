## ResNet

Directory **ResNet** contains everything required for training a **ResNet-based architecture**, where **both the RGB image and vehicle speed are used as input channels**.

- **/experiments** – Stores the experiments generated during training.
- **/utils** – Contains the utilities required by the network, including dataset generation tools and the ResNet architecture implementation.
- **run_training.sh** – Script used to launch the training process. It contains the arguments passed to the training code.
- **train_final** – Main training script implementing the full pipeline: training, validation, and testing. This script is executed through the `.sh` launcher.
- **model_eval** – Script used to evaluate a trained model in inference mode. It alternates between a **speed heatmap visualization** and a **red vehicle trail** showing the trajectory. The circuit camera must be specified.
- **log_gen_from_inference** – Script that generates a dataset and a `.log` file from model inference. This is used to compare the model’s behavior with the human driver or other trained models.
- **logs_compare_numerical** – Script that performs a **quantitative comparison between two trajectory CSV files** (typically human vs inference), comparing the positions along the trajectories.
- **logs_compare_visual** – Visual comparison between two trajectories represented by CSV files, simulating a “race” between the human and model trajectories.

### Usage

Run training:

./run_training.sh

Evaluate a model with visualization:

python3 model_eval.py --mode heatmap --cam 5

Generate inference logs:

python3 log_gen_from_inference.py --cam 5

Compare trajectories numerically:

python3 logs_compare_numerical.py --ref logs/Deepracer_BaseMap_5CCv1/dataset.csv --inf logs/infer_log_5CC.csv --plot

Visual comparison between trajectories:

python3 logs_compare_visual.py --csv_human logs/Deepracer_BaseMap_5CCv1/dataset.csv --csv_inf logs/infer_log_5CC.csv --cam 5