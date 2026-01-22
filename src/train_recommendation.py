import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the preprocessed lending data"""
    
    print("=== Loading Processed Data ===")
    data_path = Path("data/processed/lending_processed_v2.csv")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        print("Please run data_preprocessing.py first.")
        return None
    
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def prepare_features(df):
    """Prepare features for model training"""
    
    print("\n=== Preparing Features ===")
    
    # Features available at application time (what users provide)
    application_features = [
        'loan_amnt',      # User enters
        'dti',            # User provides debt/income info
        'annual_inc',     # User provides income
        'addr_state',     # User location
        'zip_code',       # User location (more granular)
        'emp_length',     # User employment history
        'loan_category',  # Derived from loan_amnt
        'dti_risk'        # Derived from dti
    ]
    
    # Filter to features that exist in the dataframe
    available_features = [f for f in application_features if f in df.columns]
    print(f"Available features for training: {available_features}")
    
    # Create feature DataFrame
    X = df[available_features].copy()
    y = df['recommended'].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def encode_categorical_features(X_train, X_test):
    """Encode categorical features"""
    
    print("\n=== Encoding Categorical Features ===")
    
    # Store encoders for later use in API
    encoders = {}
    
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    print(f"Categorical columns: {list(categorical_cols)}")
    
    for col in categorical_cols:
        print(f"Encoding {col}...")
        
        # Handle missing values
        X_train[col] = X_train[col].fillna('Unknown')
        X_test[col] = X_test[col].fillna('Unknown')
        
        # Use LabelEncoder
        le = LabelEncoder()
        
        # Fit on combined categories to ensure consistency
        all_categories = pd.concat([X_train[col], X_test[col]]).unique()
        le.fit(all_categories)
        
        # Transform
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        
        # Store encoder
        encoders[col] = le
        
        print(f"  {col}: {len(le.classes_)} unique values")
    
    return X_train, X_test, encoders

def scale_features(X_train, X_test):
    """Scale numeric features"""
    
    print("\n=== Scaling Numeric Features ===")
    
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"Scaled features shape: {X_train_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler

def train_models(X_train, y_train):
    """Train multiple models and return them"""
    
    print("\n=== Training Models ===")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"  {name} training complete!")
    
    return trained_models

def evaluate_models(models, X_train, y_train, X_test, y_test):
    """Evaluate all models and return performance metrics"""
    
    print("\n=== Evaluating Models ===")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name} Performance:")
        print("-" * 50)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Probabilities for AUC
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'auc': roc_auc_score(y_train, y_train_proba)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'auc': roc_auc_score(y_test, y_test_proba)
        }
        
        # Print metrics
        print("Training Set:")
        for metric, value in train_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        print("\nTest Set:")
        for metric, value in test_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        # Confusion Matrix
        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"  TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
        print(f"  FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
        
        # Store results
        results[name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'confusion_matrix': cm
        }
    
    return results

def select_best_model(results):
    """Select the best model based on test F1 score"""
    
    print("\n=== Selecting Best Model ===")
    
    best_model_name = None
    best_f1 = 0
    
    for name, metrics in results.items():
        test_f1 = metrics['test_metrics']['f1']
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_name = name
    
    print(f"Best model: {best_model_name}")
    print(f"Test F1 Score: {best_f1:.4f}")
    
    return best_model_name

def save_model_artifacts(model, encoders, scaler, feature_names, model_name):
    """Save the trained model and preprocessing artifacts"""
    
    print("\n=== Saving Model Artifacts ===")
    
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)
    
    # Save model
    model_file = models_path / "recommendation_model.pkl"
    joblib.dump(model, model_file)
    print(f"Model saved: {model_file}")
    
    # Save encoders
    encoders_file = models_path / "encoders.pkl"
    joblib.dump(encoders, encoders_file)
    print(f"Encoders saved: {encoders_file}")
    
    # Save scaler
    scaler_file = models_path / "scaler.pkl"
    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved: {scaler_file}")
    
    # Save feature names
    features_file = models_path / "feature_names.pkl"
    joblib.dump(feature_names, features_file)
    print(f"Feature names saved: {features_file}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'features': feature_names,
        'categorical_features': list(encoders.keys()),
        'n_features': len(feature_names)
    }
    metadata_file = models_path / "model_metadata.pkl"
    joblib.dump(metadata, metadata_file)
    print(f"Metadata saved: {metadata_file}")

def main():
    """Main training pipeline"""
    
    print("🤖 LOAN RECOMMENDATION MODEL TRAINING")
    print("=" * 60)
    
    # 1. Load data
    df = load_processed_data()
    if df is None:
        return
    
    # 2. Prepare features
    X, y = prepare_features(df)
    
    # 3. Train/test split (80/20)
    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # 4. Encode categorical features
    X_train_encoded, X_test_encoded, encoders = encode_categorical_features(
        X_train.copy(), X_test.copy()
    )
    
    # 5. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_encoded, X_test_encoded
    )
    
    # 6. Train models
    models = train_models(X_train_scaled, y_train)
    
    # 7. Evaluate models
    results = evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 8. Select best model
    best_model_name = select_best_model(results)
    best_model = models[best_model_name]
    
    # 9. Save artifacts
    save_model_artifacts(
        best_model, encoders, scaler, 
        list(X_train.columns), best_model_name
    )
    
    print("\n✅ MODEL TRAINING COMPLETE!")
    print("\nNext steps:")
    print("1. Review model performance above")
    print("2. Use the saved model in your API")
    print("3. Test predictions with sample user profiles")
    print("\nModel files saved in: models/")

if __name__ == "__main__":
    main()