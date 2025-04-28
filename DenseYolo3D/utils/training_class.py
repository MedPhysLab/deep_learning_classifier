import torch.nn as nn
import torch.optim as optim
import torch
import os
from torch.nn.utils import clip_grad_norm_
import torch
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, patience=25):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_components = {"obj_loss": 0.0, "noobj_loss": 0.0, "loc_loss": 0.0}
        total_samples = 0

        for inputs, labels, shapes in self.train_loader:
            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                print("NaN found in inputs or targets!")
                continue

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            self.optimizer.zero_grad()
            outputs = self.model(inputs, shapes)
            loss, components = self.criterion(outputs, labels)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            for k, v in components.items():
                running_components[k] += v * batch_size

        running_loss /= total_samples
        running_components = {k: v / total_samples for k, v in running_components.items()}
        return running_loss, running_components

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        running_components = {"obj_loss": 0.0, "noobj_loss": 0.0, "loc_loss": 0.0}
        total_samples = 0

        with torch.no_grad():
            for inputs, labels, shapes in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.size(0)
                total_samples += batch_size

                outputs = self.model(inputs, shapes)
                loss, components = self.criterion(outputs, labels)

                running_loss += loss.item() * batch_size
                for k, v in components.items():
                    running_components[k] += v * batch_size

        running_loss /= total_samples
        running_components = {k: v / total_samples for k, v in running_components.items()}
        return running_loss, running_components

    def train(self, num_epochs, path=None, min_improvement=1):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_components = self.train_epoch()
            val_loss, val_components = self.validate_epoch()

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} (Obj: {train_components['obj_loss']:.4f}, "
                  f"  Loc: {train_components['loc_loss']:.4f})")
            print(f"Val Loss: {val_loss:.4f} (Obj: {val_components['obj_loss']:.4f}, "
                  f"  Loc: {val_components['loc_loss']:.4f})\n")

            self.scheduler.step(val_loss)

            if path is not None:
                if val_loss < best_val_loss - min_improvement:
                    best_val_loss = val_loss
                    self.save_model(os.path.join(path, "best_model.pth"))
                    print(f"Saved best model with Val Loss: {val_loss:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    self.save_model(os.path.join(path, f"model_epoch_{epoch + 1}.pth"))
                    with open(os.path.join(path, "training_history.dat"), 'a') as f:
                        f.write(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
                                f"Val Loss: {val_loss:.4f}, "
                                f"Obj: {val_components['obj_loss']:.4f}, "
                                f"Loc: {val_components['loc_loss']:.4f}\n")

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}. Best Val Loss: {best_val_loss:.4f}")
                break

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)


