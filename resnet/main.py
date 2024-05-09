
import numpy as np
from PIL import Image # Python Imaging Library - For operations like: Image open, resize image, etc..
from IPython.display import Image as show_image  # For displaying our test images to you
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions


img = Image.open("sportscar.jpg").resize((299,299))



img = np.array(img)



img.shape


print(img.ndim)

img = img.reshape(-1,299,299,3)  


img.shape


print(img.ndim)

img = preprocess_input(img)   



incresv2_model = InceptionResNetV2(weights='imagenet', classes=1000)  


print(incresv2_model.summary())
print(type(incresv2_model))

show_image(filename='sportscar.jpg') 



preds = incresv2_model.predict(img)
print('Predicted categories:', decode_predictions(preds, top=2)[0]) 

import numpy as np

arr = np.arange(24)

arr = arr.reshape(-1,24) #We added an extra dimension to array a with reshape(-1,24)
