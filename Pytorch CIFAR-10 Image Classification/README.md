# Deep Learning Algorithms
## 2. Pytorch CIFAR-10 Image Classification
&nbsp;

### A. [Code]()
### B. [Experiment Report]()
&nbsp;


**###IMPORTANT NOTES###**  
This instruction will mainly based on Pelican servers with bash shell and Python 3.x, please feel free to use any resources you have.  
If you want to change your default shell to bash, you can do so at:  
[https://teach.engr.oregonstate.edu/teach.php?type=want_auth (Links to an external site.)](https://teach.engr.oregonstate.edu/teach.php?type=want_auth) [(Links to an external site.)](https://secure.engr.oregonstate.edu:8000/teach.php?type=want_auth)

**###Login Server###**  
Login one of the following servers:  
pelican01.eecs.oregonstate.edu  
pelican02.eecs.oregonstate.edu  
pelican03.eecs.oregonstate.edu  
pelican04.eecs.oregonstate.edu  
  
For example, if you ONID is XXXX, you can login through ssh by typing following command in your terminal:  
ssh XXX@pelican01.eecs.oregonstate.edu  
For Windows users, you can choose to use either MobaXterm or Linux Subsystem (for Windows 10 only).  
  
Once login, you can check whether GPU resources are available in this server using command:  
$ :nvidia-smi

**###Install Python Virtualenv###**  
Create a Python virtual environment:  
$ ~:virtualenv --system-site-packages --python=python3 pytorch-py3

Activate your virtualenv:  
$ ~:source ~/pytorch-py3/bin/activate  
If everything is correct, you will see there will be a "(pytorch-py3)" at the beginning of your command line  
$~:(pytorch-py3) ~:  
You will install all your Python packages related to assignment 3 in this python virtual environment  
If you want to exit, just type:  
$~:(pytorch-py3) ~:deactivate

**###Install Pytorch###**  
Make sure your Python virtual environment is activated  
Upgrade your pip:  
$~: pip install --upgrade pip

Install Pytorch and Torchvision package:

Check the version of cuda you have, and select your preference for installing:  [https://pytorch.org/get-started/locally/ (Links to an external site.)](https://pytorch.org/get-started/locally/)

Like, for Linux + pip + python + cuda 9.2:

$~:pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

After installation, you should be able to import Pytorch in your python and  torch.cuda.device_count() should return a scalar (the number of GPU you have)

torch.__version__  should return the version of torch

If your installed torch is less than version 1.4.0, you could Install tensorboardX to use tensorboard:

$~:pip install tensorboardX

If you run into cuda version problem while running the given script, you could check the installed cuda via:

$~:nvcc --version

Check the torch combined cuda:

$~:python -c 'import torch; print(torch.version.cuda)'

Then change the installed cuda environment by modifying the PATH and LD_LIBRARY_PATH variables ([https://devtalk.nvidia.com/default/topic/995815/path-amp-ld_library_path/ (Links to an external site.)](https://devtalk.nvidia.com/default/topic/995815/path-amp-ld_library_path/)):

$ ~: export PATH=/usr/local/eecsapps/cuda-9.2/bin${PATH:+:${PATH}}

$ ~: export LD_LIBRARY_PATH=/usr/local/eecsapps/cuda-9.2/lib64\{LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

You could list all the versions of installed cuda by:

$~:cd /usr/local/eecsapps/cuda  
$~:ls

**###Train Your Network###**  
Download  [cifar10_pytorch_tfboard.py](https://oregonstate.instructure.com/courses/1751431/files/77981804/download?wrap=1 "cifar10_pytorch_tfboard.py")[![Preview the document](https://oregonstate.instructure.com/images/preview.png)](https://oregonstate.instructure.com/courses/1751431/files/77981804/download?wrap=1 "Preview the document")  to your server, and run following command:  
python cifar10_pytorch_tfboard.py  
If you are using pelican and you want your code to run on a specific GPU, you can add a variable like below:  
$~: CUDA_VISIBLE_DEVICES=1 python cifar10_pytorch_tfboard.py

Above command will only allow program to use GPU 1(the second GPU on pelican). It should take less than 1 minutes to train 500 steps if you are using GPU instead of CPU.

**###Assignment Request###**  
Now open  cifar10_pytorch_tfboard.py  and briefly review what's inside, make sure you understand each line of code, and finish following tasks:

1) Add a batch normalization layer after the first fully-connected layer(fc1) (8 points).  
Save the model after training(Checkout our tutorial on how to save your model).  
Becareful that batch normalization layer performs differently between training and evalation process, make sure you understand how to convert your model between training mode and evaluation mode(you can find hints in my code).  
Observe the difference of final training/testing accuracy with/without batch normalization layer.

2) Modify our model by adding another fully connected layer with 512 nodes at the second-to-last layer (before the fc2 layer) (8 points).  
Apply the model weights you saved at step 1 to initialize to the new model(only up to fc2 layer since after that all layers are newly created) before training. Train and save the model (Hint: check the end of the assignment description to see how to partially restore weights from a pretrained weights file).

3) Try to use an adaptive schedule to tune the learning rate, you can choose from RMSprop, Adagrad and Adam (Hint: you don't need to implement any of these, look at Pytorch documentation please) (8 points).

4) Try to tune your network in another way (e.g. add/remove a layer, change the activation function, add/remove regularizer, change the number of hidden units, more batch normalization layers) not described in the previous four. You can start from random initialization or previous results as you wish (8 points).

5) Try to use the visualization toolkit, tensorboard, for tracking and visualizing your training process and include them in your report: show the loss and accuracy, visualize the model graph, and display images or other tensors as they change over time. (Hint: If you installed pytorch==1.14.0 or above, you could use torch.utils.tensorboard and the tutorials could be found at  [https://pytorch.org/docs/stable/tensorboard.html (Links to an external site.)](https://pytorch.org/docs/stable/tensorboard.html). Otherwise, you could use tensorboardX and the tutorials could be found at  [https://github.com/lanpa/tensorboardX (Links to an external site.)](https://github.com/lanpa/tensorboardX))(8 points).

For each of the settings 1) - 4), please submit a PDF report your training loss, training accuracy, validation loss and validation accuracy. Draw 2 figures for each of the settings 1) - 4) with the x-axis being the epoch number, and y-axis being the loss/accuracy, use 2 different lines in the same figure to represent training loss/validation loss, and training accuracy/validation accuracy. 5) is graded based on if the figures are from tensorboard visualization.

Name your file "firstname_lastname_hw3.pdf". Submit this pdf file on Canvas.

**###Hint for Partially Restore Weights###**  
In Pytorch trained weights are saved as *.pth file, you can load these files into a dictionary with torch.load('YOUR_FILE_NAME.pth') function. If you try to print the keys in this dictionary, you will find they are the variable name you used when you create your model.  
So you can load your pretrained weights into a dictionary 'pretrained_dict', then load your new model's weights into another dictionary 'model_dict' with method net.state_dict(). Then update weights in 'model_dict' only when keys in model_dict are also in pretrained_dict. Once you have updated model_dict, you can assign the values in this dictionary to your new model using net.load_state_dict(model_dict) method.

&nbsp;
# [Experiment Report]()