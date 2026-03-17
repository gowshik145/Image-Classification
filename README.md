# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion-MNIST dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency.

## Neural Network Model
<img width="1214" height="619" alt="Screenshot 2025-09-15 164904" src="https://github.com/user-attachments/assets/23686e71-ee9a-46fa-9c26-f9cd992e1919" />

## DESIGN STEPS

### STEP 1: Problem Statement
Define the objective of classifying fashion items (T-shirts, trousers, dresses, shoes, etc.) using a Convolutional Neural Network (CNN).

### STEP 2: Dataset Collection
Use the Fashion-MNIST dataset, which contains 60,000 training images and 10,000 test images of various clothing items.

### STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

### STEP 4: Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers to extract features and classify clothing items.

### STEP 5: Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

### STEP 6: Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.



## PROGRAM

### Name:GOWSHIK S
### Register Number: 212223220026
```python
class CNNClassifier(nn.Module):
  def __init__(self):
    super(CNNClassifier,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128*3*3,128)
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,10)

  def forward(self,x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size(0),-1)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```

```python
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
```

```python
def train_model(model,train_loader,num_epochs=3):
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    print('Name: GOWSHIK S')
    print('Register Number: 212223220026')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT
### Training Loss per Epoch
<img width="422" height="150" alt="image" src="https://github.com/user-attachments/assets/8f771369-1212-4869-8939-9aeb779858ed" />


### Confusion Matrix
<img width="808" height="707" alt="image" src="https://github.com/user-attachments/assets/18670515-8696-44cf-888f-8a1d678188ac" />


### Classification Report
<img width="552" height="373" alt="image" src="https://github.com/user-attachments/assets/168d8d9b-75a5-4840-9653-781ec3044e76" />


### New Sample Data Prediction
<img width="561" height="507" alt="image" src="https://github.com/user-attachments/assets/7f82ee73-0914-4fd9-85c3-73ae725ee80f" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
