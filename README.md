# Predicting American Sign Language
This repo has code that can predict an American Sign Language (ASL) letter.  Training the model has 90% accuracy with around 1% loss.  *I am not a data scientist*, so this model could be improved upon. 

## Model Description
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 128)       1280
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 128)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 64)        73792
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 14, 64)        36928
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 7, 25)          14425
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 3, 3, 25)          0
_________________________________________________________________
flatten (Flatten)            (None, 225)               0
_________________________________________________________________
dense (Dense)                (None, 512)               115712
_________________________________________________________________
dropout (Dropout)            (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 25)                12825
=================================================================
Total params: 254,962
Trainable params: 254,962
Non-trainable params: 0

---------------------------------------------------------------------

Epoch 1/5
428/428 [==============================] - 161s 375ms/step - loss: 2.6039 - accuracy: 0.2069 - val_loss: 0.6898 - val_accuracy: 0.7669
Epoch 2/5
428/428 [==============================] - 177s 414ms/step - loss: 0.7052 - accuracy: 0.7603 - val_loss: 0.2222 - val_accuracy: 0.9297
Epoch 3/5
428/428 [==============================] - 171s 399ms/step - loss: 0.3322 - accuracy: 0.8853 - val_loss: 0.1253 - val_accuracy: 0.9589
Epoch 4/5
428/428 [==============================] - 217s 505ms/step - loss: 0.2118 - accuracy: 0.9263 - val_loss: 0.1229 - val_accuracy: 0.9587
Epoch 5/5
428/428 [==============================] - 213s 499ms/step - loss: 0.1537 - accuracy: 0.9460 - val_loss: 0.0517 - val_accuracy: 0.9846

```

## Outcomes

