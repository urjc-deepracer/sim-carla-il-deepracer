## Project Structure

- **clientCarlaScript/**  
  Contains all scripts that directly interact with the **CARLA server**, including the code used to generate, collect, and preprocess dataset characteristics.

- **PilotNetDefault/**  
  Includes everything required for the **basic PilotNet training setup**, where **only the RGB image** is used as the input channel.

- **PilotNetEnhanced/**  
  Contains all components required for training **PilotNet with multiple inputs**, using both the **image and the vehicle speed** as input channels.

- **PilotNetEnhancedWeights/**  
  Includes the full training pipeline for **PilotNet with image and speed inputs**, incorporating **weighted loss functions** and additional strategies such as **batch mixing** to improve training robustness.

- **ResNet/**  
  Contains all necessary components for training a **ResNet-based architecture**, using **image and vehicle speed** as input channels.