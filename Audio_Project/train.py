from get_data import X_train,X_test,y_train,y_test
from model_creation import base_network,a,b,audio_a,audio_b
from keras import backend as K
from get_data import X_train, X_test, y_train, y_test
from tensorflow.keras import models,layers
from tensorflow import keras
from tensorflow.keras import *
a1=X_train[:,0]
a2=X_train[:,1]

def eucl(vects):
    z,v=vects
    return K.sqrt(K.sum(K.square(z-v),keepdims=True,axis=1))


def eucl_shape(shapes):
    s1,s2=shapes
    return (s1[0],1)
distance=layers.Lambda(eucl,eucl_shape)([a,b])
model=keras.models.Model(inputs=[audio_a,audio_b],outputs=distance)

def con(y_true,y_pred):
    margin=1
    return K.mean(y_true*K.square(y_pred)+(1-y_true)*K.square(K.maximum(1-y_pred,0)))
print(model.summary())
model.compile(loss=con,optimizer=optimizers.RMSprop())
model.fit([a1,a2],y_train,batch_size=128,epochs=13,validation_split=0.25)
model.save("audio.h5")
# model.save("model",save_format="tf")