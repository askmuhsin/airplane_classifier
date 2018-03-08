# Classify airplane images from satellite data

usage `python classifier.py`     
SVC model is saved as a pickle file, and running the above code without any    
alteration of the code will skip the training and run the classifier using the saved model.   
Instead to do the training edit the `train_model` flag to `True` in main().

---

## Classification :   
The classification is currently done using a support vector classifier.

## Dependencies
* python3+
### following python libraries :
* glob
* time
* pickle
* random
* cv2
* numpy
* matplotlib
* sklearn
