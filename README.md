# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
* For this experiment a pretrained model resnet18 is selected and transfer learning is applied on this with the dog breed classification dataset which has 133 classes. Resenet18 is selected in this case as it is a light-weight model to get started with image classification.
* Learning rate is used this depends within a range of 0.001 to 0.1, the model learning depends upon the data provided to it so used this.
* Epochs is the number of times the whole dataset is seen my model, lower epochs caused underfitting and higher epochs causes overfitting so this parameter is also useful, the ranges we gave to this are [1,2]
* Batch size is used for faster and efficient training, here 32 and 64 batch sizes are given as hyperparameters.

Remember that your README should:
- Include a screenshot of completed training jobs
Here is a screenshot of the completed training jobs
![](screenshots\trainingjobs.png)
- Logs metrics during the training process
Here are the metrics observed during training
Metrics such as training loss, validation loss, training accuracy and validaion accuracy a logged at each epoch, which can observed from the notebook in the cell after setting up training job estimator
- Tune at least two hyperparameters
In this project 3 hyperparameters learning rate, batch size, number of epochs are tuned 
- Retrieve the best best hyperparameters from all your training jobs
After hyper parameter tuning job is run on the specified ranges for each parameter the best hyperparameters retrieved are 
number of epochs - 2
batch size - 64
learning rate - 0.05979664684978815
In this case, test accuracy is taken as the evaluation metric for the getting the best hyperparameters.


## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
For model debugging the following are considered
Vanishing gradient,overfit,poor weight initialization,loss not decreasing,overtraining.
For model profiling low GPU utilization is tracked for every ten and profiler report is created with all the above attributes.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
* From the sagemaker debugger profiling report, it is very clear that majority of time is spent on training the model and next on some processing steps and very least(1%) is utilized for evaluating the model.
* In CPU operations chart data loaders consumed most of the time.
* From the report it suggests to use GPU instance for training.
* Batch size has to be increased for a better result
* In the profiler report issues are found as we are checking on GPU utilization and no gpus are found(this is expected behaviour just tried to check it)
* To gain a better accuracy we have to train this dataset on more number of epochs also have to increase the samples for unbalanced classes.

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
The model deployed is a resent model which is tuned for 2 epochs on the dog breed classification dataset. The endpoint is created in such a way that the input to it has to be in the form of bytes, and the response will have the probability of all 133 classes out of which the highest probability one can be choosen as result.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![](screenshots\endpoints.png)
## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
