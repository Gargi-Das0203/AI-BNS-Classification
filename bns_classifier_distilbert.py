"""
BNS Section Classifier using DistilBERT
=======================================

A deep learning classifier to predict the appropriate BNS (Bharatiya Nyaya Sanhita) section 
for given complaint texts using DistilBERT transformer model.

Author: Generated for BNS Classification Task
Date: September 2025
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BNSDataset(Dataset):
    """
    Custom Dataset class for BNS complaint texts and section labels.
    
    This class handles the tokenization of complaint texts and prepares them
    for training with DistilBERT model.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of complaint text strings
            labels (list): List of corresponding section IDs
            tokenizer: DistilBERT tokenizer instance
            max_length (int): Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BNSClassifier:
    """
    Main BNS Classifier class that handles data loading, model training, and evaluation.
    """
    
    def __init__(self, data_path='backend/', model_name='distilbert-base-uncased'):
        """
        Initialize the BNS Classifier.
        
        Args:
            data_path (str): Path to the directory containing CSV files
            model_name (str): HuggingFace model name for DistilBERT
        """
        self.data_path = data_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        
        # Data containers
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.bns_sections = None
        
        # Model containers
        self.model = None
        self.num_classes = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
    def load_data(self):
        """
        Load and preprocess all CSV data files.
        
        This method loads the BNS sections data and training/validation/test datasets,
        then prepares them for model training.
        """
        print("Loading data files...")
        
        # Load BNS sections information
        self.bns_sections = pd.read_csv(os.path.join(self.data_path, 'bns_sections.csv'))
        print(f"Loaded {len(self.bns_sections)} BNS sections")
        
        # Load training, validation, and test data
        self.train_data = pd.read_csv(os.path.join(self.data_path, 'train_data.csv'))
        self.val_data = pd.read_csv(os.path.join(self.data_path, 'val_data.csv'))
        self.test_data = pd.read_csv(os.path.join(self.data_path, 'test_data.csv'))
        
        print(f"Training samples: {len(self.train_data)}")
        print(f"Validation samples: {len(self.val_data)}")
        print(f"Test samples: {len(self.test_data)}")
        
        # Prepare labels - fit label encoder on all possible section IDs
        all_section_ids = self.bns_sections['section_id'].values
        self.label_encoder.fit(all_section_ids)
        self.num_classes = len(all_section_ids)
        
        print(f"Number of BNS sections to classify: {self.num_classes}")
        
    def prepare_data_loaders(self, batch_size=8, max_length=512):
        """
        Prepare PyTorch DataLoaders for training, validation, and testing.
        
        Args:
            batch_size (int): Batch size for training
            max_length (int): Maximum sequence length for tokenization
        """
        print("Preparing data loaders...")
        
        # Encode labels
        train_labels = self.label_encoder.transform(self.train_data['section_id'])
        val_labels = self.label_encoder.transform(self.val_data['section_id'])
        test_labels = self.label_encoder.transform(self.test_data['section_id'])
        
        # Create datasets
        train_dataset = BNSDataset(
            self.train_data['complaint_text'].values,
            train_labels,
            self.tokenizer,
            max_length
        )
        
        val_dataset = BNSDataset(
            self.val_data['complaint_text'].values,
            val_labels,
            self.tokenizer,
            max_length
        )
        
        test_dataset = BNSDataset(
            self.test_data['complaint_text'].values,
            test_labels,
            self.tokenizer,
            max_length
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Data loaders prepared with batch size: {batch_size}")
        
    def initialize_model(self, learning_rate=2e-5):
        """
        Initialize the DistilBERT model for sequence classification.
        
        Args:
            learning_rate (float): Learning rate for optimizer
        """
        print("Initializing DistilBERT model...")
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Calculate total training steps
        total_steps = len(self.train_loader) * 10  # Assuming 10 epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        print(f"Model initialized with {self.num_classes} output classes")
        
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
        return total_loss / len(self.train_loader)
        
    def evaluate(self, data_loader):
        """
        Evaluate the model on given data loader.
        
        Args:
            data_loader: PyTorch DataLoader for evaluation
            
        Returns:
            tuple: (average_loss, accuracy, predictions, true_labels)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels
        
    def train(self, num_epochs=10, save_path='bns_model'):
        """
        Train the BNS classifier for specified number of epochs.
        
        Args:
            num_epochs (int): Number of training epochs
            save_path (str): Path to save the trained model
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        best_val_accuracy = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Evaluate on validation set
            val_loss, val_accuracy, _, _ = self.evaluate(self.val_loader)
            
            # Store training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model(save_path)
                print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")
                
        print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")
        
    def test_model(self):
        """
        Evaluate the model on test set and provide detailed metrics.
        
        Returns:
            dict: Dictionary containing test results and metrics
        """
        print("Evaluating on test set...")
        
        test_loss, test_accuracy, predictions, true_labels = self.evaluate(self.test_loader)
        
        # Convert encoded labels back to original section IDs
        true_section_ids = self.label_encoder.inverse_transform(true_labels)
        pred_section_ids = self.label_encoder.inverse_transform(predictions)
        
        # Get section information for predictions
        results = []
        for i, (true_id, pred_id) in enumerate(zip(true_section_ids, pred_section_ids)):
            true_section = self.bns_sections[self.bns_sections['section_id'] == true_id].iloc[0]
            pred_section = self.bns_sections[self.bns_sections['section_id'] == pred_id].iloc[0]
            
            results.append({
                'complaint_text': self.test_data.iloc[i]['complaint_text'][:100] + "...",
                'true_section_id': true_id,
                'true_section_code': true_section['section_code'],
                'true_section_title': true_section['section_title'],
                'predicted_section_id': pred_id,
                'predicted_section_code': pred_section['section_code'],
                'predicted_section_title': pred_section['section_title'],
                'correct': true_id == pred_id
            })
        
        # Generate classification report
        section_names = [f"Section {sid}" for sid in self.label_encoder.classes_]
        class_report = classification_report(
            true_labels, 
            predictions, 
            target_names=section_names,
            output_dict=True
        )
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'predictions': results,
            'classification_report': class_report
        }
        
    def predict_section(self, complaint_text, top_k=3):
        """
        Predict BNS section for a given complaint text.
        
        Args:
            complaint_text (str): The complaint text to classify
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of top-k predictions with confidence scores
        """
        self.model.eval()
        
        # Tokenize input text
        encoding = self.tokenizer(
            complaint_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            section_id = self.label_encoder.inverse_transform([idx.item()])[0]
            section_info = self.bns_sections[self.bns_sections['section_id'] == section_id].iloc[0]
            
            results.append({
                'section_id': section_id,
                'section_code': section_info['section_code'],
                'section_title': section_info['section_title'],
                'section_description': section_info['section_description'],
                'crime_category': section_info['crime_category'],
                'confidence': prob.item()
            })
            
        return results
        
    def save_model(self, save_path):
        """Save the trained model and associated components."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save label encoder
        import joblib
        joblib.dump(self.label_encoder, os.path.join(save_path, 'label_encoder.pkl'))
        
        # Save BNS sections data
        self.bns_sections.to_csv(os.path.join(save_path, 'bns_sections.csv'), index=False)
        
        print(f"Model saved to {save_path}")
        
    def load_model(self, load_path):
        """Load a previously trained model."""
        import joblib
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        # Load label encoder
        self.label_encoder = joblib.load(os.path.join(load_path, 'label_encoder.pkl'))
        
        # Load BNS sections data
        self.bns_sections = pd.read_csv(os.path.join(load_path, 'bns_sections.csv'))
        
        self.model.to(device)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"Model loaded from {load_path}")
        
    def plot_training_history(self):
        """Plot training history graphs."""
        if not self.training_history['train_loss']:
            print("No training history available. Please train the model first.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.training_history['train_loss'], label='Training Loss', marker='o')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss', marker='s')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.training_history['val_accuracy'], label='Validation Accuracy', marker='o', color='green')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_data_distribution(self):
        """Analyze and visualize the data distribution across BNS sections."""
        print("Analyzing data distribution...")
        
        # Combine all data for analysis
        all_data = pd.concat([self.train_data, self.val_data, self.test_data], ignore_index=True)
        
        # Count samples per section
        section_counts = all_data['section_id'].value_counts().sort_index()
        
        # Merge with section information
        section_stats = self.bns_sections.merge(
            section_counts.reset_index().rename(columns={'index': 'section_id', 'section_id': 'sample_count'}),
            on='section_id',
            how='left'
        )
        section_stats['sample_count'] = section_stats['sample_count'].fillna(0)
        
        # Plot distribution by section
        plt.figure(figsize=(20, 8))
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(section_stats)), section_stats['sample_count'])
        plt.title('Sample Distribution Across BNS Sections')
        plt.xlabel('Section ID')
        plt.ylabel('Number of Samples')
        plt.xticks(range(len(section_stats)), section_stats['section_id'], rotation=45)
        
        # Highlight sections with very few samples
        for i, (bar, count) in enumerate(zip(bars, section_stats['sample_count'])):
            if count == 0:
                bar.set_color('red')
            elif count <= 2:
                bar.set_color('orange')
        
        # Plot distribution by crime category
        plt.subplot(2, 1, 2)
        category_counts = all_data.merge(self.bns_sections[['section_id', 'crime_category']], on='section_id')['crime_category'].value_counts()
        plt.bar(category_counts.index, category_counts.values)
        plt.title('Sample Distribution by Crime Category')
        plt.xlabel('Crime Category')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"\nData Distribution Statistics:")
        print(f"Total samples: {len(all_data)}")
        print(f"Sections with no samples: {(section_stats['sample_count'] == 0).sum()}")
        print(f"Sections with 1-2 samples: {((section_stats['sample_count'] > 0) & (section_stats['sample_count'] <= 2)).sum()}")
        print(f"Average samples per section: {section_stats['sample_count'].mean():.2f}")
        
        return section_stats

def main():
    """
    Main function to demonstrate the BNS classifier usage.
    """
    print("BNS Section Classifier using DistilBERT")
    print("=" * 50)
    
    # Initialize classifier
    classifier = BNSClassifier(data_path='backend/')
    
    # Load and analyze data
    classifier.load_data()
    classifier.analyze_data_distribution()
    
    # Prepare data for training
    classifier.prepare_data_loaders(batch_size=4, max_length=512)  # Smaller batch size due to limited data
    
    # Initialize model
    classifier.initialize_model(learning_rate=2e-5)
    
    # Train the model
    classifier.train(num_epochs=15, save_path='bns_distilbert_model')  # More epochs for small dataset
    
    # Plot training history
    classifier.plot_training_history()
    
    # Test the model
    test_results = classifier.test_model()
    
    # Display sample predictions
    print("\nSample Test Predictions:")
    print("-" * 100)
    for i, result in enumerate(test_results['predictions'][:5]):
        print(f"\nSample {i+1}:")
        print(f"Text: {result['complaint_text']}")
        print(f"True: {result['true_section_code']} - {result['true_section_title']}")
        print(f"Predicted: {result['predicted_section_code']} - {result['predicted_section_title']}")
        print(f"Correct: {result['correct']}")
    
    # Demonstrate inference on new text
    print("\nDemonstration: Predicting BNS section for new complaint")
    print("-" * 60)
    
    sample_complaint = """
    On 15th March 2025, around 10 PM, I was walking home when a group of four men 
    approached me and demanded money. When I refused, they threatened me with a knife 
    and forced me to hand over my wallet and mobile phone. They took Rs. 5000 cash 
    and my phone worth Rs. 20000 before running away.
    """
    
    predictions = classifier.predict_section(sample_complaint, top_k=3)
    
    print(f"Complaint: {sample_complaint.strip()}")
    print(f"\nTop 3 Predicted Sections:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['section_code']} - {pred['section_title']}")
        print(f"   Category: {pred['crime_category']}")
        print(f"   Confidence: {pred['confidence']:.4f}")
        print()

if __name__ == "__main__":
    main()