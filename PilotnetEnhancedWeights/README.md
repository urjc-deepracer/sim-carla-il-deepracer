## PilotNetEnhancedWeights

Directory **PilotNetEnhancedWeights** contains everything required for training **PilotNet using both the image and vehicle speed as input channels**, while also incorporating **weighted loss strategies and advanced techniques such as batch mixing** during training.

- **/experiments** – Stores the experiments generated during training.
- **/utils** – Contains the utilities required by the network, including dataset generation tools and the PilotNet architecture implementation.
- **run_weighted_training_estados.sh** – Script used to launch the weighted training process. It contains the arguments passed to the training code.
- **train_weight_estados** – Main training script implementing the full pipeline: training, validation, and testing. This script is executed through the `.sh` launcher.
- **train_weights_estados_scheduler** – Same as the previous script but with an added **learning rate scheduler** and optional **WeightedRandomSampler** support.
- **train_weights_estados_scheduler_batch_mix** – Extension of the previous approach that also applies **batch mixing during training** and balances the loss accordingly.

### Usage

Run weighted training:

./run_weighted_training_estados.sh