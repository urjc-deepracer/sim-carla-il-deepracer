## clientCarlaScript

This directory contains all scripts that directly interact with the **CARLA server**, as well as utilities used to **generate, modify, and analyse datasets**.

The project starts from a dataset organized with a structure similar to the following:

```
/datasets/
├── Deepracer_BaseMap_12C
│   
├── Deepracer_BaseMap_12CC
│   
├── Deepracer_BaseMap_12CCv2
│   
├── Deepracer_BaseMap_12Cv2
│   
├── Deepracer_BaseMap_13C
│   
├── Deepracer_BaseMap_13CC
│   ├── 
├── Deepracer_BaseMap_13CCv2
│   ├── 
├── Deepracer_BaseMap_13Cv21
│   ├── 
├── Deepracer_BaseMap_3CC
│   ├──
├── Deepracer_BaseMap_3CCv2
│   ├──
├── Deepracer_BaseMap_4C
│   ├
├── Deepracer_BaseMap_4Cv2
│   ├
├── Deepracer_BaseMap_5C
│   ├──
├── Deepracer_BaseMap_5Cv2
│   ├
├── test
│   ├── Deepracer_BaseMap_14C
│   │  
│   ├── Deepracer_BaseMap_14CC
│   │ 
│   ├── Deepracer_BaseMap_14CCv2
│   │   
│   └── Deepracer_BaseMap_14Cv2
│       
└── validation
    ├── Deepracer_BaseMap_3C
    │   
    ├── Deepracer_BaseMap_3Cv2
    │  
    ├── Deepracer_BaseMap_4CC
    │ 
    ├── Deepracer_BaseMap_4CCv2
    │  
    ├── Deepracer_BaseMap_5CC
    │   
    └── Deepracer_BaseMap_5CCv2
        


```
Each directory corresponds to a **data recording session for a specific circuit**, numbered from **3 to 14**.

Naming convention:

- **C** → Clockwise (the circuit is driven clockwise)
- **CC** → CounterClockwise (the circuit is driven in the opposite direction)
- **v2** → Second recording of the same circuit

Inside each directory there are:

- an **rgb/** folder containing RGB images  
- a **masks/** folder containing the corresponding segmentation masks  
- a **dataset.csv** file containing all metadata and the paths to each image


# Data generation and balancing

### adjust_dataset_final
Balances the dataset so that each **state (1,2,3 → left, center, right)** has the same representation across circuits.

python3 adjust_dataset_final.py --valdir ../datasets/validation/


### check_repeated_images
Checks that there are **no duplicated images or repeated samples** after preprocessing.

python3 check_repeated_images.py --base-dir ../datasets/validation/


### datasetgenfromreplayandcsvfilelaterchange
Generates a dataset directory (e.g. `Deepracer_BaseMap_4C`) from a `.log` replay file and a CSV file containing speed values.

python3 datasetgenfromreplayandcsvfilelaterchange.py


### datasetgenfromreplayandcsvfilelaterchangewithposition
Same as the previous script but also stores the **vehicle position (x, y, z)**.

python3 datasetgenfromreplayandcsvfilelaterchangewithposition.py


### delete_duplicates
Removes duplicated samples from datasets.

python3 delete_duplicates.py --base-dir ../datasets/


### delete_throttle_higher_than / delete_throttle_lower_than
Removes samples whose **throttle value is outside a defined threshold**.

python3 delete_throttle_higher_than.py


# Dataset visualization

### bin_viewer
Displays the **distribution of dataset samples in batches**.

python3 bin_viewer.py


### dataset_visualizecsv
Visualizes an entire dataset including speed, steering, RGB image and segmentation mask.

python3 dataset_visualizecsv.py --base_path ../datasets/Deepracer_BaseMap_14Cv2/


### frequency_histograms_absolute / frequency_histograms
Plots the **distribution of dataset samples**.

- `frequency_histograms_absolute` shows raw distributions for train, validation and test.
- `frequency_histograms` normalizes each dataset between 0 and 1.

python3 frequency_histograms_absolute.py


### histograms
Creates a **bar plot of state distribution (1,2,3)**.

python3 histograms.py --pattern "../datasets/validation/Deepracer_BaseMap_*/dataset.csv"


# Mask processing and visualization

### turn_black_masks
Turns the **top 100 pixels of each mask black** to remove the white walls.

python3 turn_black_masks.py --base-dir ../datasets/


### turn_black_top200_and_square_masks
Same as the previous script but also adds **200 black pixels at the top** so the final image becomes **square (800x800)**.

python3 turn_black_top200_and_square_masks.py --base-dir ../datasets/


### visualize_masks
Displays the masks of a dataset sequentially.

python3 visualize_masks.py --base-dir ../datasets/Deepracer_BaseMap_12Cv2/


# Vehicle control

### clear_vehicles
Removes all spawned vehicles from the world.

python3 clear_vehicles.py


### manualcontrol / manualcontrolspinningcam
Controls the vehicle using **WASD keys**.  
The spinningcam version rotates the camera around the vehicle.

python3 manualcontrol.py


### pdcontroller / pdcontroller30fps
PD controller used to **keep the vehicle on the center line**.

python3 pdcontroller.py


# Controller-based driving (PS4 and Switch Pro)

### datasetgenNintendoController
### datasetgenPS4Controllerjoysticks
### datasetgenPS4ControllerR2
Generate dataset samples while driving the vehicle using controllers.

python3 datasetgenNintendoController.py


### joystick_client_nintendo
### joystick_client_ps4_joysticks
### joystick_client_ps4R2
Scripts that connect to the respective controllers and send data to the dataset generation scripts.

python3 joystick_client_nintendo.py


### manualcontrolNintendoController
### manualcontrolPS4Controller
Drive the vehicle directly using controllers.

python3 manualcontrolPS4Controller.py


# Other utilities

### testtime
Compares the difference between **simulated time and real clock time**.

python3 testtime.py