# Sentimental Analysis on Audio Data

### Download it from here: [Repository](https://github.com/kushal1999seemakurthi/Deep_Learning/tree/main/projects/Sentimental_Analysis_of_Audio)

### **Aim:**
 
To classify the Emotion of the given Audio.
 
### **Description:**
 
**Data** used has 8 categories of Emotions. For the Audio processing purpose library  **Librosa** used along with *numpy* to model the data according to the requirements. **Tensorflow** framework was used to build and train DL models for the classification.
 
#### Specifications of the data and Model
 
 With asistance of Librosa Audio can be modelled into MFCCs of matrices of 55 X 44. Processed matrices could be further used as Inputs for various DL models. In this expercise models like LSTM model, CNN model, LSTM + CNN model were used.
 
* Here is the mapping for the model predictions:
 ```python
 mapping = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise']
```
