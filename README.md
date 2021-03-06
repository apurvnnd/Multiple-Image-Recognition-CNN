# Multiple-Image-Recognition-CNN

  
[Dataset](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P16-Convolutional-Neural-Networks.zip)

- The above mentioned dataset includes only two animals, Cats and Dogs. 
- I added another animal in the dataset, so you are free to choose your own.

This CNN reads for 3 animals, if you want to add more, 
**change output_dim = *desired number of animals/humans* in the defined output layer**

Program run on:
- Spyder
- Python 3.5

Libraries used: 
- Keras 
- scikit-learn 
- Tensorflow


Steps Applied : 
1. Convolution 
2. Pooling 
3. First Two Steps repeated to increase accuracy 
4. Flattening 
5. Full Connection Establishment 
6. Fitting Image to CNN


NOTE: *The program was run on windows 10, 64 bit. The epoch rate was pretty slow, to fix the issue, in fit_generator, 'steps_per_epoch' and 'validation_steps' were divied by its batch_size. It decreased the overall execution time.*
