import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)    
    
if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 100).unsqueeze(1) # Shape: [100, 1]
    y = 2 * x + 0.3 * torch.randn_like(x) # True relation: y = 2*x + noise
    
    epochs = 100
    learning_rate = 0.1
    
    optimizers_config = {
        'SGD': lambda model: optim.SGD(model.parameters(), lr=learning_rate),
        'Adam': lambda model: optim.Adam(model.parameters(), lr=learning_rate),
        'Adadelta': lambda model: optim.Adadelta(model.parameters(), lr=learning_rate),
        'Adagrad': lambda model: optim.Adagrad(model.parameters(), lr=learning_rate),
        'RMSprop': lambda model: optim.RMSprop(model.parameters(), lr=learning_rate),
    }

    loss_history = {}
    
    for opt_name, opt_func in optimizers_config.items():
        # Instantiate a new model for each optimizer to start from the same point.
        model = LinearRegressionModel()
        criterion = nn.MSELoss()
        optimizer = opt_func(model)
        history = []
        # Training loop
        for epoch in range(epochs):
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            history.append(loss.item())
        loss_history[opt_name] = history
        print(f"{opt_name} final loss: {history[-1]:.4f}")
    # Plot the loss history
    plt.figure(figsize=(10, 6))
    for opt_name, history in loss_history.items():
        plt.plot(history, label=opt_name)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Optimizer Comparison on Linear Regression")
        plt.legend()
    plt.show()
    
    # In practice, many practitioners start with Adam as a strong default.
    # Once the learning rate and training dynamics are better understood, 
    # they may switch to or compare with SGD (with momentum).
    # Ultimately, the right choice often comes down to empirical testing by 
    # monitoring convergence speed and overall validation performance.
    