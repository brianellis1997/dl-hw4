"""
Usage:
    python3 -m homework.train_planner --model mlp_planner
    python3 -m homework.train_planner --model transformer_planner
    python3 -m homework.train_planner --model cnn_planner
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .datasets.road_dataset import load_data
from .models import load_model, save_model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform_pipeline = "state_only" if args.model in ["mlp_planner", "transformer_planner"] else "default"
    
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=transform_pipeline,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    model = load_model(args.model).to(device)
    print(f"Model: {args.model}")
    
    criterion = nn.L1Loss()
    
    if args.model == "transformer_planner":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    writer = SummaryWriter(f"runs/{args.model}")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            optimizer.zero_grad()
            
            if args.model == "cnn_planner":
                predictions = model(image=batch["image"])
            else:
                predictions = model(
                    track_left=batch["track_left"],
                    track_right=batch["track_right"]
                )
            
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]
            
            masked_predictions = predictions * waypoints_mask.unsqueeze(-1)
            masked_waypoints = waypoints * waypoints_mask.unsqueeze(-1)
            
            loss = criterion(masked_predictions, masked_waypoints)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                if args.model == "cnn_planner":
                    predictions = model(image=batch["image"])
                else:
                    predictions = model(
                        track_left=batch["track_left"],
                        track_right=batch["track_right"]
                    )
                
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]
                
                masked_predictions = predictions * waypoints_mask.unsqueeze(-1)
                masked_waypoints = waypoints * waypoints_mask.unsqueeze(-1)
                
                loss = criterion(masked_predictions, masked_waypoints)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        if args.model == "transformer_planner":
            scheduler.step()
        else:
            scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = save_model(model)
            print(f"Saved best model to {save_path}")
    
    writer.close()
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                       choices=["mlp_planner", "transformer_planner", "cnn_planner"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
