# BNS Section Classifier - Setup and Usage Guide

## Overview
This project implements a deep learning classifier using DistilBERT to predict appropriate BNS (Bharatiya Nyaya Sanhita) sections for given complaint texts. The classifier works with your provided 26 BNS sections and includes OpenAI integration for enhanced performance.

## Dataset Information
- **26 BNS Sections** across 4 crime categories:
  - Offences Against Human Body
  - Offences Against Public Tranquillity  
  - Offences Against Property
  - Sexual Offences
- **Training Data**: 93 samples
- **Validation Data**: 29 samples  
- **Test Data**: 25 samples

## Files Structure
```
Mini Project 3/
├── backend/
│   ├── bns_sections.csv          # 26 BNS sections with descriptions
│   ├── train_data.csv            # Training complaint data
│   ├── val_data.csv              # Validation data
│   └── test_data.csv             # Test data
├── bns_classifier_distilbert.py  # Main classifier implementation
├── demo_enhanced_bns_classifier.py # Usage examples and demonstrations
├── openai_enhanced_bns_classifier.py # OpenAI-enhanced version
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install PyTorch (Choose based on your system)
```bash
# CPU only
pip install torch torchvision torchaudio

# GPU with CUDA (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. (Optional) Set up OpenAI API Key
For enhanced features with data augmentation:
```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac  
export OPENAI_API_KEY=your-api-key-here
```

## Quick Start

### Basic Usage
```python
from bns_classifier_distilbert import BNSClassifier

# Initialize classifier
classifier = BNSClassifier(data_path='backend/')

# Load and prepare data
classifier.load_data()
classifier.prepare_data_loaders(batch_size=4, max_length=512)
classifier.initialize_model()

# Train the model
classifier.train(num_epochs=10, save_path='bns_model')

# Make predictions
complaint = "Three men robbed me at gunpoint on the highway last night."
predictions = classifier.predict_section(complaint, top_k=3)

for i, pred in enumerate(predictions, 1):
    print(f"{i}. {pred['section_code']} - {pred['section_title']} (Conf: {pred['confidence']:.3f})")
```

### Run Complete Demo
```bash
python demo_enhanced_bns_classifier.py
```

### Run OpenAI-Enhanced Version
```bash
python openai_enhanced_bns_classifier.py
```

## Key Features

### 1. DistilBERT-Based Classification
- Uses `distilbert-base-uncased` for efficient text classification
- Optimized for complaint text analysis
- Handles variable length complaint texts (up to 512 tokens)

### 2. Comprehensive Evaluation
- Training/validation loss tracking
- Accuracy metrics and classification reports
- Confusion matrix analysis
- Sample prediction demonstrations

### 3. Data Analysis Tools
- Distribution analysis across BNS sections
- Identification of under-represented sections
- Visualization of training progress

### 4. OpenAI Integration (Enhanced Version)
- Synthetic data generation for under-represented sections
- Hybrid predictions combining DistilBERT + GPT
- Legal reasoning for predictions
- Consensus-based classification

## Model Performance and Limitations

### Expected Performance
Given the small dataset size (147 total samples), expect:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 70-85%
- **Test Accuracy**: 65-80%

### Current Limitations
1. **Small Dataset**: Only 147 total samples for 26 classes
2. **Imbalanced Data**: Some sections have 0-2 samples
3. **Limited Generalization**: May overfit to training patterns
4. **Domain Specificity**: Trained on specific complaint writing styles

### Sections with Limited/No Data
Several BNS sections have very limited training data:
- Sections with 0 samples: Need synthetic data generation
- Sections with 1-2 samples: High risk of overfitting
- Better performance expected on well-represented sections

## Improvement Strategies

### 1. Data Augmentation
```python
# Use OpenAI for synthetic data generation
from openai_enhanced_bns_classifier import OpenAIEnhancedBNSClassifier

classifier = OpenAIEnhancedBNSClassifier()
classifier.load_data()
classifier.augment_training_data(min_samples_per_section=5)
```

### 2. Advanced Training Techniques
- **Class Weighting**: Handle imbalanced data
- **Transfer Learning**: Start from legal domain models
- **Ensemble Methods**: Combine multiple models
- **Few-Shot Learning**: Better handle rare sections

### 3. External Data Sources
- Collect more real complaint data
- Use legal databases and case studies
- Implement cross-validation with legal experts
- Leverage BNS section descriptions for similarity matching

## Usage Examples

### Example 1: Murder Case
```python
complaint = """
A group of five men brutally murdered Mr. Rajesh Kumar while shouting 
communal slurs, specifically targeting him because of his religion.
"""
# Expected: Section 103(2) - Murder by group with discriminatory motive
```

### Example 2: Property Crime
```python
complaint = """
Three thieves broke into my house by opening the lock, stole jewelry 
worth Rs. 2 lakhs, and had taken precautions to hide their identity.
"""
# Expected: Section 305 - Theft in dwelling house
# Alternative: Section 330(2) - House-breaking by opening lock
```

### Example 3: Sexual Offense
```python
complaint = """
A group of men sexually assaulted a 16-year-old girl, with multiple 
attackers taking turns while others held her down.
"""
# Expected: Section 70(2) - Gang rape of minor
```

## Technical Specifications

### Model Architecture
- **Base Model**: DistilBERT (66M parameters)
- **Classification Head**: Linear layer for 26 classes
- **Max Sequence Length**: 512 tokens
- **Batch Size**: 4 (optimized for small dataset)
- **Learning Rate**: 2e-5 with linear warmup

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Linear warmup with decay
- **Epochs**: 10-15 (to prevent overfitting)
- **Gradient Clipping**: 1.0 (for stability)

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, GPU with 4GB+ VRAM
- **Training Time**: 10-30 minutes (depending on hardware)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size to 2 or 1
   - Reduce max_length to 256

2. **Poor accuracy on specific sections**
   - Check if section has sufficient training data
   - Use OpenAI augmentation for those sections

3. **Model overfitting**
   - Reduce number of epochs
   - Add dropout or other regularization

4. **OpenAI API errors**
   - Check API key is set correctly
   - Verify API quota and billing
   - Add retry logic with delays

### Performance Tips
- Use GPU if available for faster training
- Monitor validation accuracy to avoid overfitting
- Use class weights for heavily imbalanced sections
- Consider ensemble of multiple models for better accuracy

## Legal Considerations
- This model is for demonstration/research purposes
- Real legal applications require expert validation
- Predictions should be reviewed by legal professionals
- Continuous updating needed as laws evolve

## Support and Contribution
- Report issues with specific error messages
- Provide additional complaint data if available
- Suggest improvements for better accuracy
- Share results and findings for model enhancement
