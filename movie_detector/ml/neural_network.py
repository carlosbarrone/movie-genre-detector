from torch import no_grad
from torch.nn import Linear, ReLU, Sigmoid, Module
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class GenreClassifier(Module):
    def __init__(self, input_dim: int = 384, output_dim:int = 25, hidden_layer_1: int = 256, hidden_layer_2: int = 96) -> None:
        super(GenreClassifier, self).__init__()
        self.fc1 = Linear(input_dim, hidden_layer_1)
        self.relu1 = ReLU()
        self.fc2 = Linear(hidden_layer_1, hidden_layer_2)
        self.relu2 = ReLU()
        self.fc3 = Linear(hidden_layer_2, output_dim)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.sigmoid(x)


def evaluate_model(model, val_loader, criterion, threshold):
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_predictions = []
    with no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            predictions = (outputs >= threshold).int()
            all_predictions.extend(predictions.numpy())
            all_targets.extend(targets.numpy())
    avg_val_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='samples', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='samples')
    f1 = f1_score(all_targets, all_predictions, average='samples')
    model.train()
    return avg_val_loss, accuracy, precision, recall, f1


def train_model(model: GenreClassifier, criterion, optimizer, train_loader: DataLoader, val_loader: DataLoader, epochs=100, patience:int=10, alpha: float = 0.5, threshold: float =0.5) -> GenreClassifier:
    best_compound_score = float('-inf')
    patience_counter = 0
    best_model_state = None
    beta = 1 - alpha

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_val_loss, accuracy, precision, recall, f1 = evaluate_model(model, val_loader, criterion, threshold)
        normalized_loss = 1 / (1 + avg_val_loss)
        compound_score = alpha * normalized_loss + beta * f1
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {avg_val_loss:.4f}, F1: {f1}, Compound Score: {compound_score}')

        if compound_score > best_compound_score:
            best_compound_score = compound_score
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Best model restored.")
    return model
