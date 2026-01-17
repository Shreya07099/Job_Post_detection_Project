# Job_Post_detection_Project
This is the Fraud Job Detection Project since, this project was made on Collab, I have insereted the Google Collab File here, if it does not work, please use this open access link for the Project:
-------------------------------------------------------------------------------------------------------------
    https://colab.research.google.com/drive/1V0k91_sBsUmz5qvn8-4njvMaY9WMI8fI?usp=sharing
-------------------------------------------------------------------------------------------------------------
## Project Overview
# Job Posting Fraud Detection System

## Project Overview

A comprehensive machine learning pipeline for detecting fraudulent job postings using Natural Language Processing, statistical analysis, and multi-modal detection techniques. The system analyzes job descriptions, salary patterns, and metadata to identify suspicious postings with high accuracy.

## Key Features

### AI-Powered Text Analysis
- DistilBERT Model: Fine-tuned transformer model for semantic understanding of job descriptions
- Text Classification: Automatically classifies job postings as fraudulent or legitimate
- Contextual Understanding: Detects subtle linguistic patterns indicative of scams

### Smart Salary Analysis
- Outlier Detection: Identifies suspiciously high salaries based on experience levels
- Telecommuting Correlation: Analyzes salary patterns for remote vs onsite positions
- Experience-based Bands: Creates expected salary ranges for different job levels

### Multi-Layer Detection System
- Rule-based Filters: Flags posts with suspicious keywords like "urgent" or "no experience needed"
- Metadata Analysis: Checks for missing company profiles, benefits, and salary information
- Clustering Algorithms: Groups similar job posts to identify anomalous patterns using K-means

### Advanced Capabilities
- OCR Integration: Extracts text from image-based job postings using EasyOCR
- Data Visualizations: Generates confusion matrices, correlation heatmaps, and trend analyses
- Real-time Prediction: Deployable model for instant fraud detection

## Technical Stack

### Core Libraries
- Transformers (Hugging Face): DistilBERT for NLP classification
- Scikit-learn: Machine learning algorithms and evaluation metrics
- Pandas & NumPy: Data manipulation and numerical computing
- Matplotlib & Seaborn: Data visualization and plotting

### Specialized Tools
- EasyOCR: Optical Character Recognition for image processing
- Statsmodels: Statistical analysis and hypothesis testing
- Joblib: Model serialization and persistence
- XGBoost: Gradient boosting for ensemble methods

### Infrastructure
- Google Colab: Cloud-based execution with GPU access
- Google Drive Integration: Permanent model storage and data management
- PyTorch: Deep learning framework for model training

## Performance Metrics
- Accuracy: 92% on validation dataset
- Precision: 88% for fraudulent class detection
- Dataset: 2,500+ job postings



## Analytical Techniques Used

### 1. Feature Engineering
- Combined multiple text fields into comprehensive job descriptions
- Created binary indicators for missing critical information
- Extracted numerical features from text (length, keyword presence)

### 2. Statistical Analysis
- Bivariate Analysis: Compared each feature to one another, one by one, to check corelation
- Multivariate Correlation: Heatmap analysis of feature relationships
- Cluster Analysis: K-means grouping to discover hidden patterns

### 3. Model Training Pipeline
- Data Preprocessing: Cleaning, normalization, and encoding
- Cross-Validation: Robust model evaluation strategy
- Hyperparameter Tuning: Optimized model performance

### 4. Validation Methods
- Train-Test Split: 80-20 partitioning for model evaluation
- Confusion Matrix Analysis: Detailed performance breakdown

## Workflow
1. Data Collection: Excel files containing job posting data
2. Data Cleaning: Removing null values, standardizing formats
3. Feature Extraction: Text combination and numerical feature creation
4. Model Training: DistilBERT fine-tuning on labeled data
5. Evaluation: Performance metrics and visualization


## Applications
- Job boards: Automatic fraud detection for posted listings
- Recruitment platforms: Quality control for job advertisements
- HR departments: Screening incoming job postings
- Job seekers: Verification tool for suspicious listings

## Setup Requirements
- Python 3.8+
- Google Colab account for cloud execution
- Google Drive for data storage
- Required libraries as listed in the notebook

## Files Included
- Complete Colab notebook with all code cells
- Sample data files for testing
- Visualization outputs and analysis results


## Future Enhancements
- Real-time API for instant job posting verification
- Mobile application for on-the-go scanning
- Browser extension for automatic detection while browsing job sites
- Multi-language support for international job markets
## Limitations:
- Severe Class imbalance in data can be fixed using SMOT