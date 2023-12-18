# VGG-11-on-Cifar10-using-PyTorch
This project is to understand how distributed training of machine learning (ML)  algorithms is performed using PyTorch and Torch Serve.
The file contains the pyspark script that implements distributed programming by using 
pytorch. 
Below are the details and instructions for the same: 
Folder1: Google collab code – Runtime type- T4 GPU
Part1_part2googlecollab
Code: Part1and2.ipynb
 Part1and2.py
 40minibatches_part1and 2.ipynb
 40minibatches_part1and 2.py

Part2_approch2and PART3_vscode/ pytorch
Input Files: Data downloaded from the script given, CIFAR10
Requirements: 
The code can be run in any environment with the respective installations.
Python, spark, Apache spark. If you are trying locally, you must be Apache spark installed and 
configured.
The provided code is created on Google collab. Each cell needs to be executed one by one to 
download the packages and created the instant runtime. 
Input : Run the cell in google colab to download the dataset
Program: Running the cells one by one and execute the code in one manner. 
Output: Since it is parallel programming, it takes time to run. 
For VScode: Install the dependencies from https://pytorch.org/serve/getting_started.htm for 
torchseve part 3 and install jdk , java 11 . 
Set the environment variables : export JAVA_HOME=/path/to/your/java
Run main.py part1 and main_dist.py for part2 and torchserve.py for part3
-----------------------------------------------------------END----------------------------------------------------------
