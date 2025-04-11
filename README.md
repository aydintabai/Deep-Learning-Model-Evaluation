# A Comparison of CNN Variations for Image Classification

**Aydin Tabatabai**  
University of California, San Diego  
atabatabai@ucsd.edu

<center>
<strong>Aydin Tabatabai</strong><br>
University of California, San Diego<br>
atabatabai@ucsd.edu
</center>

**Abstract**

Convolutional neural networks are widely used for image classification tasks, but their performance can depend heavily on architectural choices and training strategies. This project investigates how design choices in network architecture, activation functions, optimizers, etc. affect performance of classification on a comprehensive dataset. Seven variations of models were implemented and trained from scratch, including a baseline CNN, variants of the baseline, multiple ResNet18 variants, and VGG11. All of the models were trained and evaluated over ten epochs, measuring test accuracy, training time, and final training loss. The model that was able to achieve the highest accuracy was ResNet18 with the Adam optimizer with a test accuracy at 76.57%. The next most accurate models were a LeakyReLU variant of ResNet18 and then a deeper baseline CNN. However, in contrast, VGG11 performed very poorly, possibly due to an architectural mismatch or insufficient tuning. The results demonstrate that deeper architectures and adaptive optimizers offer meaningful improvements when configured appropriately. These findings highlight the importance of selecting model components and training setups carefully when designing learning systems for image classification.

**Introduction**

Convolutional neural networks (CNNs) are widely used for solving image classification tasks because of their ability to learn visual patterns directly from raw image data. The utility and adaptability of these networks, however, are not only determined by the presence of convolutional layers. Factors like model layer depth, choice of activation function, use of regularization techniques, choice of optimization algorithm, etc. can all influence how effectively a network learns and generalizes to new data.

This report looks at how those choices impact performance of a model when training a CNN on CIFAR-10 \[1\].  CIFAR is an image classification dataset that contains 60,000 color images across 10 different categories. For this study, seven models were built and tested, all with different variations of design. Three of the variations included a basic baseline CNN, a deeper version with additional layers, and a version that added batch normalization and dropout. Then also, several versions of ResNet18 \[2\] architecture were also tested, each using different combinations of activation functions and optimizers. Finally, VGG11 \[3\] was tested as well, as a deeper, feedforward alternative. All models were trained under the same conditions and analyzed based on test accuracy, final training loss, and total training time. 

The purpose of this study is to highlight the trade offs between these depth, regularization, optimization, etc. strategies and choices in neural network design. By doing so, insight can be gained into which factors most significantly contribute to effective learning and generalization in image classification. The results will help to show the strengths of each approach and offer useful takeaways for building effective models in similar real-world situations.

**Method**

**Architectures**  
Seven CNN models were designed to evaluate the effects of architectural choices, activation functions, and optimization strategies on classification performance. The baseline model consisted of a basic CNN with two convolutional layers followed by two fully connected layers. To measure the impact of depth, a deeper CNN variant was created with four convolutional layers. Then, another variation of the baseline network was created that incorporated batch normalization and dropout to determine the effect of regularization.

In addition to these models that were custom built, predefined architectures were also explored. ResNet18 was one architecture selected, testing it with three different variations. First, the original ResNet18 using the ReLU activation function and Adam optimizer was trained. Then, a second version using stochastic gradient descent (SGD) instead of Adam was done. Finally, a third was trained with LeakyReLU activation instead of ReLU. Additionally, a VGG11 model was also included to assess performance of a different architecture. All of the models concluded with a fully connected layer producing ten output classes, which corresponds to the CIFAR-10 dataset.

**Dataset**  
All of the models were trained and evaluated on the CIFAR-10 dataset, which contains 60,000 color images across ten categories, with 50,000 images used for training and 10,000 for testing. The ten classes are airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Before training, each image was converted to a tensor and normalized using a mean and standard deviation of (0.5, 0.5, 0.5) for each color channel.

**Training Procedure**  
Each model was trained for 10 epochs using the framework PyTorch. A batch size of 64 was used for both training and testing. Most models were trained using the Adam optimizer with a learning rate of 0.001. For comparison, one variant of ResNet18 was trained using SGD with a learning rate of 0.001 and momentum of 0.9. All models used cross-entropy loss as the objective function. All training was conducted on a T4 GPU.

**Evaluation Metrics**  
Model performance was measured using three primary metrics, test accuracy, training time, and final training loss. Test accuracy measured the percentage of correct predictions on the CIFAR-10 test set. Training time was recorded in seconds for each model over 10 epochs to assess computational cost. Final training loss reflected the cross-entropy loss value at the end of the last epoch, giving insight into convergence behavior. These metrics allowed for a thorough and comprehensive comparison of the different models across multiple aspects.

**Experiment**

To evaluate the effect of architectural and training variations on image classification performance, the seven CNN models were evaluated using test accuracy, training time, and final training loss. The model experiments included a baseline CNN, a regularized CNN with batch normalization and dropout, a deeper CNN with four convolutional layers, several variants of ResNet18, and VGG11. All models were trained from scratch for 10 epochs using the same evaluation metrics.

| Model | Params | Activation | Optimizer | Accuracy (%) | Training Time (s) | Final Loss |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Baseline CNN | 545098 | ReLU | Adam | 72.11	 | 149.94 | 0.2155 |
| Baseline CNN (BN \+ Dropout) | 545290 | ReLU | Adam | 71.49 | 145.76 | 0.9842 |
| Deeper Baseline CNN | 307786 | ReLU | Adam | 75.26 | 149.65	 | 0.1985 |
| ResNet18 | 11181642 | ReLU | Adam | 76.57	 | 242.34	 | 0.2174 |
| VGG11 | 128807306 | ReLU | Adam | 10.00 | 618.86	 | 2.3027 |
| ResNet18 (SGD) | 11181642 | ReLU | SGD | 64.96	 | 233.09 | 0.2288 |
| ResNet18 (LeakyReLU) | 11181642 | LeakyReLU | Adam | 75.82	 | 245.02 | 0.2153 |

After training, the baseline CNN, which consisted of two convolutional layers and two fully connected layers, achieved a test accuracy of 72.11% with a final training loss of 0.2155. To compare performance, a variant of this baseline model was trained that incorporated batch normalization and dropout. This model, however, achieved a slightly lower accuracy of 71.49% and significantly higher training loss of 0.9842, suggesting that the regularization may have hindered convergence. In contrast, a deeper version of the baseline model, which had four convolutional layers, improved performance to 75.26% and achieved the lowest final loss among all models of 0.1985, showing the benefits of increasing layer depth of the model for feature extraction.

The ResNet18 architecture was able to achieve the highest test accuracy overall at 76.57%, showing the effectiveness of residual connections for training deeper models. Changing the standard activation function of ReLU to LeakyReLU resulted in similar performance of 75.82% test accuracy, illustrating that activation choice had a slight impact. However, when ResNet18 was trained with SGD instead of Adam, the test accuracy dropped to 64.96%, despite having a similar training time. This result highlights the importance of optimizer choice when training networks.

![][image1]  
VGG11, however, performed significantly worse than all other models, achieving only 10.00% test accuracy with a final loss of 2.3027. This outcome suggests that the architecture failed to learn under the current training configuration, likely because the model was not well suited for the dataset or training conditions used, since VGG11 was designed for larger images or the learning rate was not tuned enough.

The learning of each model can also be further understood by comparing their training loss over time. Most models showed a steady decrease in loss with the Deeper Baseline CNN and ResNet variants converging quicker than others. Differently, VGG11’s training loss remained flat with a slight decrease in the first two epochs, supporting that it failed to train.

![][image2]

Overall, the most accurate and consistent performers were ResNet18 (ReLU), ResNet18 (LeakyReLU), and the Deeper Baseline CNN, all achieving over 75% test accuracy. The results also illustrate that while increasing model complexity can be helpful, it must also have the right optimization and training strategies to be successful.

**Conclusion**

This project explored the effects of architectural choices, activation functions, and optimization strategies on the performance of convolutional neural networks trained on the CIFAR-10 image classification dataset. A total of seven models were implemented and compared, ranging from a simple baseline CNN to more complex architectures such as ResNet18 and VGG11. Each model was trained under the same conditions and evaluated using test accuracy, training time, and final training loss.

Among all of the experiments, ResNet18 with ReLU and the Adam optimizer achieved the highest test accuracy at 76.57%, closely followed by ResNet18 with LeakyReLU and the Deeper Baseline CNN. These results illustrate that deeper architectures and residual connections contribute significantly to model performance, especially when paired with adaptive optimizers like Adam. In contrast, the SGD variant of ResNet18 showed a noticeable drop in performance, highlighting the significance of optimizer choice when training from scratch.

The baseline model performed relatively well with a test accuracy of 72.11%. However, the deeper version clearly showed how additional convolutional layers impact the models performance with the deeper version attaining a test accuracy of 75.26%. This supports that adding depth helps extract features better. Additionally, the addition of batch normalization and dropout to the baseline CNN did not result in higher accuracy, but it increased training loss. On the other hand, VGG11 severely underperformed, likely due to a mismatch between the model design and the dataset.

Overall, the study demonstrated that architectural decisions, such as model depth, optimization strategies, etc., play a critical role in training effectiveness and generalization. While in situations more complex models can lead to better results, they must be implemented with the right training settings for its use case. These results help show how to build and improve CNNs for image classification problems.

**References**  
\[1\] Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images* (Technical  
Report). University of Toronto.  
\[2\] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition.  
arXiv. https://arxiv.org/abs/1512.03385  
\[3\] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for Large-Scale  
image recognition. *Computer Vision and Pattern Recognition*. http://export.arxiv.org/pdf/1409.1556