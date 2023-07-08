# Overview:
In this project I examined images of roots taken of switchgrass grown in a hydroponic setting.  The goal was to classify the roots as either "hairy", or "not hairy," as hairy roots are more likely to survive to maturity. Hydroponic grow systems are becoming more viable prevalent, and systems such as this can help improve the overall yields of the process.


## The Process:
The architectures tested were, AlexNet, LeNet, VGGnet, and SimpleNet.  Each layer of each architecture was constructed using the PyTorch library.  Feature Engineering was pretty straight forward with this dataset.  Each image was vectorized using the CV2 library, then converted to the size each network required to function properly.  The time to train each network was recorded, and not surprisingly, the larger networks like VGGNet and AlexNet had long training times, where as SimpleNEt and LeNet had fairly short training times.  However, SimpleNet and LeNEt had fairly high accuracy on this dataset as well.  AlexNEt and VGGNEt did do better, getting accuracies at 99 and 100%, but at the expense of a steep training time
