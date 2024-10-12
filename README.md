# Thesis-project_endVersion

In the following it is described how to run this project

## Tools used
- Python 3.8
- Carla 0.9.13
- Nengo 3.2.0
- NengoLoihi 1.1.0
- Nest 2.12.0
- Tensorflow 2.11.0
- KerasSpiking 0.3.0
- Operating System: Ubuntu 20.04

## How to run the project
The navigation project is inside the Folder **R-STDP**
- Run the Carla Simulator first
- For running the Nengo Network
	- Run nengo_controller.py
- For running the NestNetwork
	- Run nest_controller.py
- If you want to generate LiDAR labels during the run, set the attribute **generateLidarLabels** = True
- If you want to use the LiDAR-based FCN/FCN-SNN then set the attribute **useFCN** = True, otherwise the ground Truth in CARLA is used
	- Currently the default is set to FCN-SNN network when the  **useFCN** = True in environment.py (between Line 67-72), comment/uncomment the Line for switching between the FCN/FCN-SNN approaches
- There are two possible scenarios: Set the attribute **scenario** to either "Scenario1" or "Scenario2"

## FCN Network
The fcn-network is located inside the **fcn-for-lane-detection** folder. The dataset used for training is also provided there. Run **LoDNN_pooling.py** inside the folder **road_detection** for training or evaluating the FCN.
## FCN-SNN Network
The fcn-network is located inside the **fcn-for-lane-detection-snn** folder. The dataset used for training is also provided there. Run **SNN.py** inside the folder **road_detection** for training or evaluating the FCN.

## LiDAR_Data
The folder **lidar_data** contains the generated training labels by CARLA and **cropData.py** is used to crop the images into a 400x200 for the LiDAR images and 50x25 for the ground truth labels.