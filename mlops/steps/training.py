import os
import sys
import argparse
import pandas as pd
import torch
from movie_detector.ml.neural_network import GenreClassifier, train_model
from torch import tensor, float32
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.nn import BCELoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--hidden-layer-1", type=int, default=256)
    parser.add_argument("--hidden-layer-2", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=100)
    args, _ = parser.parse_known_args()
    print('Loading data.')
    X_train_df = pd.read_csv(os.path.join(args.train,'X_train.csv'))
    y_train_df = pd.read_csv(os.path.join(args.train,'y_train.csv'))
    X_val_df = pd.read_csv(os.path.join(args.validation,'X_val.csv'))
    y_val_df = pd.read_csv(os.path.join(args.validation,'y_val.csv'))

    movie_titles_y_train = y_train_df.pop('title')
    movie_titles_y_val = y_val_df.pop('title')

    X_train_tensor = tensor(X_train_df.values, dtype=float32)
    y_train_tensor = tensor(y_train_df.values, dtype=float32)
    X_val_tensor = tensor(X_val_df.values, dtype=float32)
    y_val_tensor = tensor(y_val_df.values, dtype=float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    input_dim = X_train_df.shape[1]
    num_labels = y_train_df.shape[1]
    print('Loading complete.')
    model = GenreClassifier(
        input_dim=input_dim,
        output_dim=num_labels,
        hidden_layer_1=args.hidden_layer_1,
        hidden_layer_2=args.hidden_layer_2
    )

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print('Starting training.')
    train_model(
        model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        threshold=args.threshold
    )
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    print('Training complete.')
    sys.exit(0)
