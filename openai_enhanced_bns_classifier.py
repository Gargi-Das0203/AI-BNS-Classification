"""
OpenAI-Enhanced BNS Classifier
==============================

This enhanced version leverages OpenAI's API for data augmentation and improved predictions.
It addresses the small dataset limitation by generating synthetic training data.
"""

import openai
import os
from bns_classifier_distilbert import BNSClassifier
import pandas as pd
import json
from typing import List, Dict
import time
import random

class OpenAIEnhancedBNSClassifier(BNSClassifier):
    """
    Enhanced BNS Classifier that uses OpenAI API for data augmentation and prediction improvement.
    """
    
    def __init__(self, data_path='backend/', model_name='distilbert-base-uncased', openai_api_key=None):
        """
        Initialize the enhanced classifier with OpenAI integration.
        
        Args:
            data_path (str): Path to data files
            model_name (str): DistilBERT model name
            openai_api_key (str): OpenAI API key (can also be set as env variable)
        """
        super().__init__(data_path, model_name)
        
        # Set up OpenAI API
        if openai_api_key:
            openai.api_key = openai_api_key
        elif 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        else:
            # Use provided API key as default
            openai.api_key = ""
            print("Using provided OpenAI API key for enhanced features.")
    
    def generate_synthetic_complaints(self, section_info: Dict, num_complaints: int = 5) -> List[str]:
        """
        Generate synthetic complaint texts for a specific BNS section using OpenAI.
        
        Args:
            section_info (dict): Dictionary containing BNS section information
            num_complaints (int): Number of synthetic complaints to generate
            
        Returns:
            List[str]: List of generated complaint texts
        """
        if not openai.api_key:
            print("OpenAI API key not available. Skipping synthetic data generation.")
            return []
        
        prompt = f"""
        Generate {num_complaints} realistic police complaint texts that would fall under the following BNS (Bharatiya Nyaya Sanhita) section:
        
        Section: {section_info['section_code']}
        Title: {section_info['section_title']}
        Description: {section_info['section_description']}
        Crime Category: {section_info['crime_category']}
        
        Requirements:
        1. Each complaint should be written as if filed by a victim or witness
        2. Include specific details like dates, times, locations, and circumstances
        3. The complaints should clearly match the legal definition of this section
        4. Use realistic Indian names, places, and contexts
        5. Vary the writing style and complexity
        6. Each complaint should be 200-500 words long
        
        Format: Return only the complaint texts, separated by "---COMPLAINT---"
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal expert helping generate realistic police complaint texts for training a legal AI system."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.8
            )
            
            # Parse the generated complaints
            content = response.choices[0].message.content
            complaints = [c.strip() for c in content.split("---COMPLAINT---") if c.strip()]
            
            print(f"Generated {len(complaints)} synthetic complaints for {section_info['section_code']}")
            return complaints
            
        except Exception as e:
            print(f"Error generating synthetic complaints: {e}")
            return []
    
    def augment_training_data(self, min_samples_per_section: int = 5, max_new_per_section: int = 10):
        """
        Augment training data by generating synthetic complaints for under-represented sections.
        
        Args:
            min_samples_per_section (int): Minimum samples each section should have
            max_new_per_section (int): Maximum new samples to generate per section
        """
        if not openai.api_key:
            print("OpenAI API key not available. Cannot augment training data.")
            return
        
        print("Starting data augmentation with OpenAI...")
        
        # Analyze current data distribution
        current_counts = self.train_data['section_id'].value_counts()
        
        augmented_data = []
        
        for _, section in self.bns_sections.iterrows():
            section_id = section['section_id']
            current_count = current_counts.get(section_id, 0)
            
            if current_count < min_samples_per_section:
                needed = min(min_samples_per_section - current_count, max_new_per_section)
                
                print(f"Generating {needed} complaints for {section['section_code']} (current: {current_count})")
                
                # Generate synthetic complaints
                synthetic_complaints = self.generate_synthetic_complaints(section.to_dict(), needed)
                
                # Add to augmented data
                for complaint in synthetic_complaints:
                    augmented_data.append({
                        'section_id': section_id,
                        'complaint_text': complaint,
                        'section_code': section['section_code'],
                        'section_title': section['section_title'],
                        'section_description': section['section_description'],
                        'crime_category': section['crime_category'],
                        'synthetic': True
                    })
                
                # Add delay to respect API rate limits
                time.sleep(1)
        
        if augmented_data:
            # Create augmented DataFrame
            augmented_df = pd.DataFrame(augmented_data)
            
            # Add synthetic data to training set
            original_size = len(self.train_data)
            
            # Create map_id and complaint_id for synthetic data
            max_map_id = self.train_data['map_id'].max() if not self.train_data.empty else 0
            max_complaint_id = self.train_data['complaint_id'].max() if not self.train_data.empty else 0
            
            augmented_df['map_id'] = range(max_map_id + 1, max_map_id + 1 + len(augmented_df))
            augmented_df['complaint_id'] = range(max_complaint_id + 1, max_complaint_id + 1 + len(augmented_df))
            
            # Combine with original training data
            self.train_data = pd.concat([self.train_data, augmented_df], ignore_index=True)
            
            print(f"Training data augmented: {original_size} → {len(self.train_data)} samples")
            
            # Save augmented training data
            self.train_data.to_csv(os.path.join(self.data_path, 'train_data_augmented.csv'), index=False)
            print("Augmented training data saved to train_data_augmented.csv")
        
        else:
            print("No data augmentation needed or possible.")
    
    def get_openai_prediction(self, complaint_text: str, top_k: int = 3) -> List[Dict]:
        """
        Use OpenAI to predict BNS sections for a complaint text.
        
        Args:
            complaint_text (str): The complaint text to classify
            top_k (int): Number of top predictions to return
            
        Returns:
            List[Dict]: List of predictions with reasoning
        """
        if not openai.api_key:
            print("OpenAI API key not available. Using local model only.")
            return self.predict_section(complaint_text, top_k)
        
        # Create prompt with BNS section information
        sections_info = ""
        for _, section in self.bns_sections.iterrows():
            sections_info += f"{section['section_code']}: {section['section_title']}\n"
        
        prompt = f"""
        As a legal expert in Indian law, analyze the following police complaint and determine which BNS (Bharatiya Nyaya Sanhita) section(s) it falls under.
        
        Available BNS Sections:
        {sections_info}
        
        Complaint Text:
        "{complaint_text}"
        
        Please:
        1. Identify the top {top_k} most applicable BNS sections
        2. For each section, provide:
           - Section code (e.g., "Section 103(1)")
           - Confidence level (0.0 to 1.0)
           - Brief reasoning (1-2 sentences)
        
        Format your response as JSON:
        {{
            "predictions": [
                {{
                    "section_code": "Section XXX",
                    "confidence": 0.XX,
                    "reasoning": "Brief explanation"
                }}
            ]
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal expert specializing in Indian criminal law and BNS sections."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Enrich with section details
            enriched_predictions = []
            for pred in result['predictions']:
                section_code = pred['section_code']
                section_info = self.bns_sections[self.bns_sections['section_code'] == section_code]
                
                if not section_info.empty:
                    section_data = section_info.iloc[0]
                    enriched_predictions.append({
                        'section_id': section_data['section_id'],
                        'section_code': section_code,
                        'section_title': section_data['section_title'],
                        'section_description': section_data['section_description'],
                        'crime_category': section_data['crime_category'],
                        'confidence': pred['confidence'],
                        'reasoning': pred['reasoning'],
                        'source': 'OpenAI'
                    })
            
            return enriched_predictions
            
        except Exception as e:
            print(f"Error with OpenAI prediction: {e}")
            print("Falling back to local model prediction...")
            return self.predict_section(complaint_text, top_k)
    
    def hybrid_prediction(self, complaint_text: str, top_k: int = 3) -> Dict:
        """
        Combine local DistilBERT model and OpenAI predictions for better accuracy.
        
        Args:
            complaint_text (str): The complaint text to classify
            top_k (int): Number of top predictions to return
            
        Returns:
            Dict: Combined predictions with both local and OpenAI results
        """
        print("Getting hybrid prediction (Local + OpenAI)...")
        
        # Get local model predictions
        local_predictions = self.predict_section(complaint_text, top_k)
        for pred in local_predictions:
            pred['source'] = 'Local DistilBERT'
        
        # Get OpenAI predictions
        openai_predictions = self.get_openai_prediction(complaint_text, top_k)
        
        # Combine and analyze
        result = {
            'complaint_text': complaint_text,
            'local_predictions': local_predictions,
            'openai_predictions': openai_predictions,
            'consensus': [],
            'disagreement': []
        }
        
        # Find consensus and disagreements
        local_sections = {pred['section_code'] for pred in local_predictions}
        openai_sections = {pred['section_code'] for pred in openai_predictions}
        
        consensus_sections = local_sections.intersection(openai_sections)
        
        for section in consensus_sections:
            local_pred = next(p for p in local_predictions if p['section_code'] == section)
            openai_pred = next(p for p in openai_predictions if p['section_code'] == section)
            
            result['consensus'].append({
                'section_code': section,
                'section_title': local_pred['section_title'],
                'local_confidence': local_pred['confidence'],
                'openai_confidence': openai_pred['confidence'],
                'average_confidence': (local_pred['confidence'] + openai_pred['confidence']) / 2
            })
        
        # Note disagreements
        disagreement_sections = local_sections.symmetric_difference(openai_sections)
        for section in disagreement_sections:
            if section in local_sections:
                pred = next(p for p in local_predictions if p['section_code'] == section)
                result['disagreement'].append({
                    'section_code': section,
                    'section_title': pred['section_title'],
                    'source': 'Local only',
                    'confidence': pred['confidence']
                })
            else:
                pred = next(p for p in openai_predictions if p['section_code'] == section)
                result['disagreement'].append({
                    'section_code': section,
                    'section_title': pred['section_title'],
                    'source': 'OpenAI only',
                    'confidence': pred['confidence']
                })
        
        return result

def demonstrate_openai_features():
    """
    Demonstrate the OpenAI-enhanced features of the BNS classifier.
    """
    print("=== OpenAI-Enhanced BNS Classifier Demo ===\n")
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  No OpenAI API key found in environment variables.")
        print("To use OpenAI features, set your API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nWill proceed with local model features only.\n")
    
    # Initialize enhanced classifier
    classifier = OpenAIEnhancedBNSClassifier(data_path='backend/')
    
    # Load data
    classifier.load_data()
    
    # Option 1: Data Augmentation (if OpenAI key available)
    if openai.api_key:
        print("1. DATA AUGMENTATION WITH OPENAI")
        print("-" * 40)
        
        # Show current data distribution
        print("Current data distribution:")
        stats = classifier.analyze_data_distribution()
        
        # Ask user if they want to augment data
        augment = input("\nDo you want to augment training data using OpenAI? (y/n): ").lower().strip()
        
        if augment == 'y':
            classifier.augment_training_data(min_samples_per_section=3, max_new_per_section=5)
        else:
            print("Skipping data augmentation.")
    
    # Prepare for training/prediction
    classifier.prepare_data_loaders(batch_size=4, max_length=512)
    classifier.initialize_model()
    
    # Option 2: Hybrid Predictions
    print("\n2. HYBRID PREDICTION DEMONSTRATION")
    print("-" * 40)
    
    sample_complaints = [
        "A group of 6 men from upper caste murdered Mr. Kumar, a Dalit, while "
        "shouting casteist slurs and saying he should know his place in society.",
        
        "Three thieves broke into my house at night, opened the lock, and stole "
        "jewelry worth Rs. 3 lakhs. They had planned everything carefully."
    ]
    
    for i, complaint in enumerate(sample_complaints, 1):
        print(f"\nSample {i}: {complaint}")
        print("-" * 50)
        
        if openai.api_key:
            # Get hybrid prediction
            hybrid_result = classifier.hybrid_prediction(complaint, top_k=2)
            
            print("LOCAL MODEL PREDICTIONS:")
            for j, pred in enumerate(hybrid_result['local_predictions'], 1):
                print(f"  {j}. {pred['section_code']} - {pred['section_title']} (Conf: {pred['confidence']:.3f})")
            
            print("\nOPENAI PREDICTIONS:")
            for j, pred in enumerate(hybrid_result['openai_predictions'], 1):
                print(f"  {j}. {pred['section_code']} - {pred['section_title']} (Conf: {pred['confidence']:.3f})")
                print(f"      Reasoning: {pred['reasoning']}")
            
            if hybrid_result['consensus']:
                print("\nCONSENSUS (Both models agree):")
                for cons in hybrid_result['consensus']:
                    print(f"  ✓ {cons['section_code']} - Avg Confidence: {cons['average_confidence']:.3f}")
            
            if hybrid_result['disagreement']:
                print("\nDISAGREEMENTS:")
                for disagr in hybrid_result['disagreement']:
                    print(f"  ⚠ {disagr['section_code']} - {disagr['source']} (Conf: {disagr['confidence']:.3f})")
        
        else:
            # Use local model only
            predictions = classifier.predict_section(complaint, top_k=2)
            print("LOCAL MODEL PREDICTIONS:")
            for j, pred in enumerate(predictions, 1):
                print(f"  {j}. {pred['section_code']} - {pred['section_title']} (Conf: {pred['confidence']:.3f})")

def main():
    """
    Main function to run the OpenAI-enhanced BNS classifier demonstration.
    """
    print("OPENAI-ENHANCED BNS SECTION CLASSIFIER")
    print("=" * 50)
    print("This enhanced version uses OpenAI API for:")
    print("• Data augmentation for under-represented sections")
    print("• Improved prediction accuracy through hybrid approach")
    print("• Legal reasoning for predictions")
    print("=" * 50)
    
    demonstrate_openai_features()
    
    print("\n" + "=" * 50)
    print("ENHANCEMENT BENEFITS:")
    print("✓ Addresses small dataset limitation through synthetic data")
    print("✓ Provides reasoning for predictions")
    print("✓ Combines transformer efficiency with LLM accuracy")
    print("✓ Enables consensus-based predictions")
    print("=" * 50)

if __name__ == "__main__":
    main()