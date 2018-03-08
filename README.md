# Classify airplane images from satellite data

usage `python svm_classifier.py`     
SVC model is saved as a pickle file, and running the above code without any    
alteration of the code will skip the training and run the classifier using the saved model.   
Instead to do the training edit the `train_model` flag to `True` in main().

---

usage `python cnn_classifier.py`     
A LeNet like implementation. Running the code above will commence the training.    
![lenet](https://github.com/askmuhsin/airplane_classifier/blob/master/images/lenet_sch.png)

---

## Dependencies
* python3+
### following python libraries :
* keras
* glob
* time
* pickle
* random
* cv2
* numpy
* matplotlib
* sklearn
