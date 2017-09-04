# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
 
 ## AWS for Training
 1. Use Spot Instance
    * EC2 Dashboard -> Spot Request -> Request Spot Instances
    * Create new key pair-> key pair name: carnd1
    * Create new security group": 
       security group name: jupyter 
       inbound: ssh, port 22
       custom tcp: port 8888
    * AMI : "Select", search community AMIs: 	udacity-carnd-advanced-deep-learning
    * Instance Type: "Select": "Instance Type": GPU instances: g34xlarge
    * "Next"
    * Set Keypair and role: -> key pair name: carnd1
    * Manage Firewall rules: -> jupyter
    * "Review"
    * "Launch"
 2. Connect AWS Instance
    * EC2 Dashboard -> Instances
    * Right click the instance: "connect"
    * copy the connection string
    * In terminal:
       ssh -i "carnd1.pem" ubuntu@ec2-xxx-xxx-xxx-xxx.us-west-1.compute.amazonaws.com -L 8888:127.0.0.1:8888
       # clone the project
       git clone https://github.com/MichaelTien8901/CarND-Semantic-Segmentation.git
       cd CarND-Semantic-Sementation
       jupyter notebook
       
    * Now use browser to open 127.0.0.1:8888 for jupyter notebook. We can then use it to editor python files.
    * Run program. In another terminal, 
      ssh -i "carnd1.pem" ubuntu@ec2-xxx-xxx-xxx-xxx.us-west-1.compute.amazonaws.com
      
      cd CarND-Semantic-Sementation
      python main.py
      
       
       
    
 
    
 
