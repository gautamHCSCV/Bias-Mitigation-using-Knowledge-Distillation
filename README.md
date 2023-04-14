# Bias-Mitigation-using-Knowledge-Distillation

Reproduced the results on cifar-10 dataset using EfficientNet-B0. </br>
Test Accuracy is 0.9142

![image](https://user-images.githubusercontent.com/65457437/232093938-9e9a05bb-88dd-4313-baff-6a58da3aa648.png)

From the results we can observe that some classes have higher recall (upto 0.97) while
some have low recall (upto 0.84). Similar trends are observed for other performance
metrics (precision and F1-score). It is most likely an algorithmic bias.
Algorithmic bias can occur when the machine learning model is biased towards certain
features or patterns in the data that are more prevalent in some classes than others. For
example, if the model is trained on images of cats and dogs, and the dog images have more
distinct features, the model may be more biased towards classifying images as dogs even if
the image is a cat.
It is also possible that the model is overfitting to some classes during training, leading to a
better performance on those classes during testing. This can happen if the model is too
complex or the training data is too small relative to the number of model parameters.

## Co-advise: Cross Inductive Bias Distillation

First we train a teacher model (EfficientNet-B4). Once the
teacher model is trained, its knowledge is transferred to a student model that is trained on
the target domain. The teacher model's predictions on the source domains are used as soft
targets for training the student model on the target domain. The idea is that the teacher
model's knowledge of the common features across the source domains will help the student
model learn features that are more transferable to the target domain, even if the target
domain has different biases and distributions than the source domains.

Results of teacher model (EfficientNet-B4) trained on CIFAR-10:

![image](https://user-images.githubusercontent.com/65457437/232094843-26338715-8e7f-4a55-b3eb-e4144913b993.png)

Performance of student model (EfficientNet-B0) after knowledge distillation:

![image](https://user-images.githubusercontent.com/65457437/232094933-a0a08045-c084-4207-a280-f5a71cf51a69.png)

</br></br>
Disparate Impact (DI) of original model: 0.9376 </br>
Disparate Impact (DI) after bias mitigation: 0.9669 </br>
Higher DI corresponds to lower bias. Hence we can clearly see that there is reduction in
bias and we performed bias mitigation successfully.
</br></br>
The bias mitigation technique suggested in the paper “Co-advise: Cross Inductive Bias
Distillation” works best for our case. EfficientNet-B0 is a state-of-the-art model and
provides a low on test set of CIFAR-10 dataset.
The paper provides the best approach for bias mitigation. First we train a teacher model
(EfficientNet-B4). Once the teacher model is trained, its knowledge is transferred to a
student model that is trained on the target domain. The teacher model's predictions on the
source domains are used as soft targets for training the student model on the target
domain. The idea is that the teacher model's knowledge of the common features across the
source domains will help the student model learn features that are more transferable to the
target domain, even if the target domain has different biases and distributions than the
source domains.

