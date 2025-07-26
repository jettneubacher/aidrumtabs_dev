import torch
import torch.optim as optim
from model import DrumHitModel
from dataset import load_data
from torch.optim.lr_scheduler import StepLR

# FOCAL LOSS IMPLEMENTATION
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=True, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce_loss = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Load Data and Initialize Model
train_loader, test_loader = load_data('data/enst_drums/final_npz.npz')

input_shape = train_loader.dataset[0][0].shape
num_classes = train_loader.dataset[0][1].shape[0]
model = DrumHitModel(input_shape, num_classes)

# Loss, Optimizer, Scheduler
criterion = FocalLoss(alpha=1.0, gamma=2.0, logits=True)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

# Training Function
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            if batch_idx == 0:
                with torch.no_grad():
                    print(f"\n[DEBUG][Epoch {epoch+1}] First batch predictions:")
                    print("Raw logits:", outputs[0])
                    print("Sigmoid outputs:", torch.sigmoid(outputs[0]))
                    print("Ground truth:", labels[0])

                    pred_sigmoid = torch.sigmoid(outputs)
                    avg_pred = pred_sigmoid.mean(dim=0)
                    print("Average prediction per class:", avg_pred)

                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            print(f"{name} grad mean: {param.grad.mean():.6f}, std: {param.grad.std():.6f}")

                    if torch.isnan(loss):
                        print("NaN loss detected!")

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        with torch.no_grad():
            print(f"Logits range after epoch {epoch+1}: min={outputs.min().item():.2f}, max={outputs.max().item():.2f}")
            print("Mean abs logit:", torch.abs(outputs).mean().item())

        scheduler.step()
        validate_model(model, test_loader)

# Validation Function
def validate_model(model, test_loader, threshold=0.4):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > threshold).float()
        matches = (predicted == labels).all(dim=1)
        accuracy = matches.sum().item() / len(labels) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")

# Train and Save Model
train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10)
torch.save(model.state_dict(), "models/model1.pth")
print("Model saved to 'models/model1.pth'")
