# Deep Learning Algorithms
## 1. CIFAR-10 Image Classification using Fully Connected Neural Network
&nbsp;

### A. [Code]()
### B. [Experiment Report]()
&nbsp;

In this assignment, you are going to implement a one hidden layer fully connected neural network using Python from the given skeleton code  [mlp_skeleton.py](https://oregonstate.instructure.com/courses/1751431/files/77004411/download?wrap=1 "MLP_Skeleton.py")[![Preview the document](https://oregonstate.instructure.com/images/preview.png)](https://oregonstate.instructure.com/courses/1751431/files/77004411/download?wrap=1 "Preview the document")  on Canvas (find in the Files tab). This skeleton code forces you to write linear transformation, ReLU, sigmoid cross-entropy layers as separate classes. You can add to the skeleton code as long as you follow its class structure. Given N training examples in 2 categories  ![\{(\mathbf{x}_1, y_1),\ldots,(\mathbf{x}_N,y_N)\}](https://oregonstate.instructure.com/equation_images/%255C%257B%2528%255Cmathbf%257Bx%257D_1%252C%2520y_1%2529%252C%255Cldots%252C%2528%255Cmathbf%257Bx%257D_N%252Cy_N%2529%255C%257D "\{(\mathbf{x}_1, y_1),\ldots,(\mathbf{x}_N,y_N)\}"){  (  x  1  ,  y  1  )  ,  …  ,  (  x  N  ,  y  N  )  }, your code should implement backpropagation using the cross-entropy loss (see Assignment 1 for the formula) on top of a sigmoid layer: (e.g.  ![p(c_1|x)=\frac{1}{1+\exp(-f(x))}, p(c_2|x)=\frac{1}{1+\exp(f(x))}](https://oregonstate.instructure.com/equation_images/p%2528c_1%257Cx%2529%253D%255Cfrac%257B1%257D%257B1%2B%255Cexp%2528-f%2528x%2529%2529%257D%252C%2520p%2528c_2%257Cx%2529%253D%255Cfrac%257B1%257D%257B1%2B%255Cexp%2528f%2528x%2529%2529%257D "p(c_1|x)=\frac{1}{1+\exp(-f(x))}, p(c_2|x)=\frac{1}{1+\exp(f(x))}")p  (  c  1  |  x  )  =  1  1  +  exp  ⁡  (  −  f  (  x  )  )  ,  p  (  c  2  |  x  )  =  1  1  +  exp  ⁡  (  f  (  x  )  )), where you should train for an output  ![f(x)=\mathbf{w}_{2}^\top g(\mathbf{W}_1^\top \mathbf{x}+b)+c](https://oregonstate.instructure.com/equation_images/f%2528x%2529%253D%255Cmathbf%257Bw%257D_%257B2%257D%255E%255Ctop%2520g%2528%255Cmathbf%257BW%257D_1%255E%255Ctop%2520%255Cmathbf%257Bx%257D%2Bb%2529%2Bc "f(x)=\mathbf{w}_{2}^\top g(\mathbf{W}_1^\top \mathbf{x}+b)+c")f  (  x  )  =  w  2  ⊤  g  (  W  1  ⊤  x  +  b  )  +  c.  ![LaTeX: g\left(x\right)=\max\left(x,0\right)](https://oregonstate.instructure.com/equation_images/g%255Cleft(x%255Cright)%253D%255Cmax%255Cleft(x%252C0%255Cright) "g\left(x\right)=\max\left(x,0\right)")g  (  x  )  =  max  (  x  ,  0  ) is the ReLU activation function (note Assignment #1 used a sigmoid activation but here it's ReLU),  ![\mathbf{W}_1](https://oregonstate.instructure.com/equation_images/%255Cmathbf%257BW%257D_1 "\mathbf{W}_1")W  1 is a matrix with the number of rows equal to the number of hidden units, and the number of columns equal to the input dimensionality.

**Finish the above project and write a report (in pdf) with following questions:**

**Please put the report(in pdf) and the source code into a same zip file, "firstname_lastname_hw2.zip". Submit this zip file on Canvas. You have to make sure your code could run and produce reasonable results!**

1) Write a function that evaluates the trained network (5 points), as well as computes all the subgradients of  ![W_1](https://oregonstate.instructure.com/equation_images/W_1 "W_1")W  1 and  ![W_2](https://oregonstate.instructure.com/equation_images/W_2 "W_2")W  2 using backpropagation (5 points).

2) Write a function that performs stochastic mini-batch gradient descent training (5 points). You may use the deterministic approach of permuting the sequence of the data. Use the momentum approach described in the course slides.

3) Train the network on the attached 2-class dataset extracted from CIFAR-10: (data can be found in the  [cifar-2class-py2.zip](https://oregonstate.instructure.com/courses/1751431/files/77004417/download?wrap=1 "cifar-2class-py2.zip")  file on Canvas.). The data has 10,000 training examples in 3072 dimensions and 2,000 testing examples. For this assignment, just treat each dimension as uncorrelated to each other. Train on all the training examples, tune your parameters (number of hidden units, learning rate, mini-batch size, momentum) until you reach a good performance on the testing set. What accuracy can you achieve? (20 points based on the report).

(4) Training Monitoring: For each epoch in training, your function should evaluate the training objective, testing objective, training misclassification error rate (error is 1 for each example if misclassifies, 0 if correct), testing misclassification error rate (5 points).

(5) Tuning Parameters: please create three figures with following requirements. Save them into jpg format:  
i) test accuracy with different number of batch size  
ii)test accuracy with different learning rate  
iii) test accuracy with different number of hidden units

(6) Discussion about the performance of your neural network.

**[Check here for some tips on debugging this assignment!](https://oregonstate.instructure.com/courses/1751431/discussion_topics/8769709 "Assignment #2 Tips")**

&nbsp;
# [Experiment Report]()