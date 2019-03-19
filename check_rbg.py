from PIL import Image     
import os       
path = 'C:/TensorFlow/research/object_detection/images/test/' 
for file in os.listdir(path):      
     extension = file.split('.')[-1]
     if extension == 'jpg':
           fileLoc = path+file
           img = Image.open(fileLoc)
           if img.mode != 'RGB':
                 print(file+', '+img.mode)