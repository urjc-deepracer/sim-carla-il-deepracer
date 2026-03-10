## PilotNetEnhanced

Directory **PilotNetEnhanced** contains everything required for training **PilotNet using both the image and vehicle speed as input channels**.

- **/experiments** – Stores the experiments generated during training.
- **/utils** – Contains the utilities required by the network, including dataset generation tools and the PilotNet architecture implementation.
- **run_training.sh** – Script used to launch the training process. It contains the arguments passed to the training code.
- **train_final** – Main training script implementing the full pipeline: training, validation, and testing. This script is executed through the `.sh` launcher.
- **run_carla_autopilot_rgb** – Script used to test a trained model in inference mode inside CARLA. The visualization uses a third-person videogame-style camera perspective.
- **fancyvideocam** – Script used to test a trained model in inference. It alternates between a **dynamic heatmap visualization** and a **red trajectory trail** following the vehicle. The circuit camera must be specified.
- **inference_img** – Script that performs inference on a single image (typically a mask). It receives the image and speed as input and outputs **throttle and steer**.
- **log_gen_from_inference** – Generates a dataset and `.log` file from model inference. This is used later for comparison with the human driver or other models.
- **logs_compare_numerical**, **logs_compare_numerical_speed**, **logs_compare_numerical_speed_by_states** – Scripts that perform **quantitative comparisons** between two trajectory CSV files (typically human vs inference). They compare positions, speeds by position, and error metrics by driving state.
- **logs_compare_visual** – Visual comparison between two trajectories represented by CSV files. It simulates a “race” between the human and model trajectories.
- **model_eval** – Script used to evaluate a trained model in CARLA. It alternates between **speed heatmap visualization** and a **vehicle trail visualization**.

### Usage

Run training:

./run_training.sh

Test a trained model in CARLA (third-person camera):

python3 run_carla_autopilot_rgb.py

Run inference visualization with heatmap:

python3 fancyvideocam.py --mode heatmap --cam 3

Run inference on a single image:

python3 inference_img.py --img ../../imagen2.png --model experiments/exp_debug_1769708013/trained_models/pilot_net_model_best_123.pth --speed 0.888

Generate logs from inference:

python3 log_gen_from_inference.py --cam 5

Compare trajectories numerically:

python3 logs_compare_numerical.py --ref logs/Deepracer_BaseMap_5CCv1/dataset.csv --inf logs/infer_log_5CC.csv --plot

Visual comparison between trajectories:

python3 logs_compare_visual.py --csv_human logs/Deepracer_BaseMap_5CCv1/dataset.csv --csv_inf logs/infer_log_5CC.csv --cam 5

Evaluate model with visualization:

python3 model_eval.py --mode heatmap --cam 5