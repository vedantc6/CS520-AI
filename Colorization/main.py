import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('Images/scene1.jpeg') 
img.show() 
print(np.asarray(img).shape)