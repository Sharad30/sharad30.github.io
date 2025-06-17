# Gradient Vanishing Analysis in VGG19

### Motivation

Neural networks tend to learn very complex patterns in the data and we can leverage this by having networks with very deep layers. But having this power is like a double edged sword, where just stacking layers on top of each other is not going to ensure greater accuracy and we might stumble upon a problem called as `Vanishing Gradients`.

`Vanishing Gradients` problem takes away the power to learn from the data, after a point in time. In short the gradients which play a key role during backpropogation, they start to vanish (very close to 0), due to which weight optimization does not happen.

So in this notebook we wish to demonstrate this practically with CNNs and we first start with VGG19 network and understand how the gradients vary across different layers.

### Approach

Here we track and compares **average gradient norms per epoch** across multiple layers in VGG19 to observe potential vanishing gradients.

The intuition to choose these different layers across all the 19 layers in VGG is that, we want to have a broad sense of how gradients vary in each section of network i.e when the network just starts to learn in first few layers, when network is slightly deep enough to learn some features, when the network is very deep enough to capture nuanced features and the final layers which are used to generate classification probabilities.

This is will give us a better understanding how gradients behave in each of these sections.

#### Layers tracked:
- `features[0]`: Early convolutional layer
- `features[10]`: Mid-level convolutional layer
- `features[28]`: Deep convolutional layer
- `classifier[0]`: First fully connected layer

**Import libraries**


```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import gaussian_kde
from matplotlib.animation import FuncAnimation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Load VGG19 pretrained model**


```python
# Load pretrained VGG19 and modify classifier
vgg19 = models.vgg19(pretrained=True)
vgg19.classifier[6] = nn.Linear(4096, 10)

for module in vgg19.modules():
    if isinstance(module, nn.ReLU):
        module.inplace = False
        
vgg19 = vgg19.to(device)
```

**Verify no. of parameters to be trained**


```python
trainable_params = sum(p.numel() for p in vgg19.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
```

    Trainable parameters: 139,611,210


**Register backward hook to log gradients**


```python
# Gradient storage
epoch_gradients = {
    'conv1': [], 'conv_mid': [], 'conv_deep': [], 'fc1': []
}
batch_grads = {'conv1': [], 'conv_mid': [], 'conv_deep': [], 'fc1': []}

# Hook function
def register_hook(module, name):
    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            batch_grads[name].append(grad_output[0].norm().item())
    module.register_full_backward_hook(hook)

# Register hooks
register_hook(vgg19.features[0], 'conv1')
register_hook(vgg19.features[10], 'conv_mid')
register_hook(vgg19.features[28], 'conv_deep')
register_hook(vgg19.classifier[0], 'fc1')
```

**Load CIFAR10 data and apply trasnformations**


```python
# CIFAR-10 loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset = datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

    100%|██████████| 170M/170M [00:07<00:00, 24.3MB/s] 


**Define loss and optimizer**


```python
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg19.parameters(), lr=1e-4)
```

**Train the model and randomly log 20 gradients for each epoch per layer**


```python
# Train for 10 epochs and log average gradient norms
for epoch in tqdm(range(10)):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg19(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # For each layer, sample 20 values from collected gradients
    for layer in batch_grads:
        grads = batch_grads[layer]
        if len(grads) >= 20:
            sampled = np.random.choice(grads, 20, replace=False)
        else:
            sampled = np.pad(grads, (0, 20 - len(grads)), constant_values=0)
        epoch_gradients[layer].append(sampled)
        batch_grads[layer].clear()

    print(f"Epoch {epoch+1} complete")
```

    100%|██████████| 10/10 [30:00<00:00, 180.03s/it]

    Epoch 10 complete


    



```python

layers = ['conv1', 'conv_mid', 'conv_deep', 'fc1']
```


```python
def create_log_hist_animation_matplotlib(epoch_gradients, layers, num_epochs=10, save_path='log_hist_gradients.gif'):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Log-spaced bins from 1e-10 to 1e0 (1)
    log_bins = np.logspace(-10, 0, 50)

    def update(epoch):
        ax.clear()
        for layer in layers:
            data = epoch_gradients[layer][epoch]
            if len(data) > 1 and np.isfinite(data).all():
                log_data = np.abs(data) + 1e-10  # Avoid log(0)
                hist, bins = np.histogram(log_data, bins=log_bins, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                ax.plot(bin_centers, hist, label=layer)

        ax.set_xscale('log')
        ax.set_xlim(1e-10, 1)
        ax.set_ylim(0, None)
        ax.set_title(f"Log-Scaled Gradient Histogram - Epoch {epoch + 1}")
        ax.set_xlabel("Gradient Magnitude (log scale)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, which="both", ls="--", linewidth=0.5)

    ani = FuncAnimation(fig, update, frames=num_epochs, repeat=False)
    ani.save(save_path, writer='pillow', fps=1)
    print(f"Saved animation to {save_path}")

```


```python
create_log_hist_animation_matplotlib(epoch_gradients, layers, num_epochs=10, save_path='log_hist_gradients_vgg19.gif')
```

    Saved animation to log_hist_gradients_vgg19_fps20.gif
    


![Histograms](assets/vgg19/log_hist_gradients_vgg19.gif)

**Observations**

- The graph shows how the distribution of gradient magnitudes (on a log scale) evolves across epochs for each layer.
- As we can see the for each of the considered layers, the gradient magnitude lies between 1e-3 to 1, which is still on the higher side and the gradients haven't vanished.
- This could be happening because VGG19 has 19 layers and it is still not deep enough for any of the layers to vanish just yet (keeping in mind that the above network was only trained for 10 epochs).
- To further extend our experiment lets increase the number of layers in VGG19 and see if this behavior stays put.
