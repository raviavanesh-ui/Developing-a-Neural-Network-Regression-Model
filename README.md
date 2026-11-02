# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Avanesh.R

### Register Number: 212225240018

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Load Dataset
dataset1 = pd.read_csv('data.csv')

X = dataset1[['Input']].values
y = dataset1[['Output']].values


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)


# Normalize Input
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert to Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Neural Network
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.history = {'loss': []}

    def forward(self, x):
        return self.model(x)


# Initialize Model
ai_brain = NeuralNet()

criterion = nn.MSELoss()

optimizer = optim.Adam(
    ai_brain.parameters(),
    lr=0.01
)


# Training Function
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):

    for epoch in range(1, epochs + 1):

        optimizer.zero_grad()

        output = ai_brain(X_train)

        loss = criterion(output, y_train)

        loss.backward()

        optimizer.step()

        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


# Train Model
train_model(
    ai_brain,
    X_train_tensor,
    y_train_tensor,
    criterion,
    optimizer
)


# Test Model
with torch.no_grad():

    prediction_test = ai_brain(X_test_tensor)

    test_loss = criterion(prediction_test, y_test_tensor)

    print(f'Test Loss: {test_loss.item():.6f}')


# Plot Loss
loss_df = pd.DataFrame(ai_brain.history)

loss_df.plot()

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


# Prediction for Input 9
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)

X_n1_1_scaled = scaler.transform(X_n1_1.numpy())

prediction = ai_brain(
    torch.tensor(X_n1_1_scaled, dtype=torch.float32)
).item()

print(f'Prediction: {prediction}')
```

### Dataset Information
<img width="157" height="377" alt="Screenshot 2026-11-02 100039" src="https://github.com/user-attachments/assets/c7887831-0c17-4aa1-a7bb-f3348ef70cd1" />

### OUTPUT

### Training Loss Vs Iteration Plot

<img width="577" height="650" alt="Screenshot 2026-11-02 100454" src="https://github.com/user-attachments/assets/a049e3a2-74da-4d70-953b-3447766bf17e" />

### New Sample Data Prediction

<img width="292" height="15" alt="Screenshot 2026-11-02 100433" src="https://github.com/user-attachments/assets/c37b17ed-7a71-4168-b257-f32a610ffee0" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
