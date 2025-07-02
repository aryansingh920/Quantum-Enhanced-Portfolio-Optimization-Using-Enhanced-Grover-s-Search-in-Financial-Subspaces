import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SVMRegimeDetector:
    """
    SVM-based regime detection system for financial markets.
    Supports multiple kernels and hyperparameter optimization via Grid Search.
    Enhanced with proper future regime prediction capabilities.
    """

    def __init__(self, cv_folds=5, test_size=0.2, random_state=42, prediction_mode='future'):
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.prediction_mode = prediction_mode  # 'current' or 'future'
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.grid_search = None
        self.feature_names = None
        self.df = None

    def load_data(self, csv_path):
        """Load and prepare data from CSV file."""
        print("📊 Loading data...")
        self.df = pd.read_csv(csv_path)

        # Convert Date column to datetime
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)

        print(
            f"✅ Data loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        print(f"📈 Date range: {self.df.index.min()} to {self.df.index.max()}")

        # Display regime distribution
        if 'regime' in self.df.columns:
            regime_counts = self.df['regime'].value_counts()
            print(f"🎯 Regime distribution:\n{regime_counts}")

        return self.df

    def prepare_features(self, feature_set='scaled'):
        """
        Prepare feature matrix and target vector.
        
        Args:
            feature_set: 'scaled', 'raw', or 'all'
        """
        print(
            f"🔧 Preparing features (set: {feature_set}, mode: {self.prediction_mode})...")

        # Define feature columns based on set selection
        if feature_set == 'scaled':
            feature_cols = [
                col for col in self.df.columns if col.endswith('_scaled')]
        elif feature_set == 'raw':
            feature_cols = ['close', 'high', 'low', 'open', 'volume',
                            'ma5', 'ma20', 'rsi', 'returns', 'volatility']
        elif feature_set == 'all':
            feature_cols = [col for col in self.df.columns if col != 'regime']
        else:
            raise ValueError("feature_set must be 'scaled', 'raw', or 'all'")

        # Filter existing columns
        feature_cols = [col for col in feature_cols if col in self.df.columns]

        if not feature_cols:
            raise ValueError("No valid feature columns found!")

        self.feature_names = feature_cols
        print(f"📋 Selected features: {feature_cols}")

        # Prepare features and target
        X = self.df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        y = self.df['regime'].fillna(method='ffill')

        # 🔥 KEY MODIFICATION: Shift labels for future prediction
        if self.prediction_mode == 'future':
            print("🚀 Shifting labels for FUTURE regime prediction...")
            y = y.shift(-1)  # Shift labels back by 1 day
            # Drop the last row since it won't have a future regime
            X = X.iloc[:-1]
            y = y.iloc[:-1]
            print("✅ Labels shifted: Features from day t → predict regime at t+1")

        # Remove any remaining NaN rows
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"✅ Features prepared: {X.shape}")
        print(f"🏷️  Unique regimes: {list(self.label_encoder.classes_)}")

        return X, y_encoded

    def split_data(self, X, y, method='random'):
        """
        Split data into train/test sets.
        
        Args:
            method: 'random' or 'temporal'
        """
        print(f"✂️  Splitting data ({method})...")

        if method == 'random':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state,
                stratify=y
            )
        elif method == 'temporal':
            # Use temporal split for time series data
            split_idx = int(len(X) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            raise ValueError("method must be 'random' or 'temporal'")

        print(f"📊 Train set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def setup_grid_search(self, kernel_types=['rbf', 'linear', 'poly']):
        """
        Setup comprehensive grid search parameters.
        
        Args:
            kernel_types: List of kernel types to test
        """
        print("⚙️  Setting up Grid Search parameters...")

        # Comprehensive parameter grid
        param_grid = []

        # RBF kernel parameters
        if 'rbf' in kernel_types:
            param_grid.append({
                'svm__kernel': ['rbf'],
                'svm__C': [0.1, 1, 10, 100, 1000],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'svm__class_weight': [None, 'balanced']
            })

        # Linear kernel parameters
        if 'linear' in kernel_types:
            param_grid.append({
                'svm__kernel': ['linear'],
                'svm__C': [0.1, 1, 10, 100, 1000],
                'svm__class_weight': [None, 'balanced']
            })

        # Polynomial kernel parameters
        if 'poly' in kernel_types:
            param_grid.append({
                'svm__kernel': ['poly'],
                'svm__C': [0.1, 1, 10, 100],
                'svm__degree': [2, 3, 4],
                'svm__gamma': ['scale', 'auto', 0.01, 0.1],
                'svm__class_weight': [None, 'balanced']
            })

        # Sigmoid kernel parameters
        if 'sigmoid' in kernel_types:
            param_grid.append({
                'svm__kernel': ['sigmoid'],
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.01, 0.1],
                'svm__class_weight': [None, 'balanced']
            })

        total_combinations = sum(
            len(p['svm__C']) *
            len(p.get('svm__gamma', [1])) *
            len(p.get('svm__degree', [1])) *
            len(p['svm__class_weight'])
            for p in param_grid
        )
        print(f"🔍 Grid search will test {total_combinations} combinations")

        return param_grid

    def train_model(self, X_train, y_train, kernel_types=['rbf', 'linear', 'poly'],
                    cv_method='standard', scoring='accuracy', n_jobs=-1):
        """
        Train SVM model with grid search optimization.
        
        Args:
            cv_method: 'standard' or 'timeseries'
            scoring: Scoring metric for optimization
            n_jobs: Number of parallel jobs
        """
        print("🚀 Starting model training with Grid Search...")

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=self.random_state, probability=True))
        ])

        # Setup cross-validation
        if cv_method == 'timeseries':
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
            print(f"📈 Using TimeSeriesSplit with {self.cv_folds} folds")
        else:
            cv = self.cv_folds
            print(f"🔄 Using standard {self.cv_folds}-fold CV")

        # Setup grid search
        param_grid = self.setup_grid_search(kernel_types)

        self.grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring=scoring,
            n_jobs=n_jobs, verbose=1, return_train_score=True
        )

        # Train model
        print("⏳ Training in progress...")
        self.grid_search.fit(X_train, y_train)

        self.best_model = self.grid_search.best_estimator_

        print("✅ Training completed!")
        print(f"🏆 Best parameters: {self.grid_search.best_params_}")
        print(f"📊 Best CV score: {self.grid_search.best_score_:.4f}")

        return self.best_model

    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation."""
        print("📈 Evaluating model performance...")

        if self.best_model is None:
            raise ValueError("Model not trained yet!")

        # Predictions
        y_pred = self.best_model.predict(X_test)

        # Only get probabilities if available
        try:
            y_pred_proba = self.best_model.predict_proba(X_test)
        except AttributeError:
            y_pred_proba = None
            print("⚠️  Probability predictions not available for this model")

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)

        print(f"🎯 Test Accuracy: {accuracy:.4f}")

        # Classification report
        regime_names = [str(name) for name in self.label_encoder.classes_]
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=regime_names))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=regime_names, yticklabels=regime_names)
        plt.title('Confusion Matrix - Regime Detection')
        plt.ylabel('True Regime')
        plt.xlabel('Predicted Regime')
        plt.tight_layout()
        plt.show()

        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }

    def predict_regime_for_date(self, target_date, feature_set='scaled'):
        """
        🎯 Predict regime for a specific date.
        
        Args:
            target_date: String (YYYY-MM-DD) or datetime object
            feature_set: Feature set to use for prediction
            
        Returns:
            dict: Prediction results with regime, probability, and date info
        """
        if self.best_model is None:
            raise ValueError(
                "Model not trained yet! Call train_model() first.")

        if self.df is None:
            raise ValueError("No data loaded! Call load_data() first.")

        # Convert target_date to datetime if it's a string
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)

        print(
            f"🎯 Predicting regime for date: {target_date.strftime('%Y-%m-%d')}")

        # Determine the date to use for features
        if self.prediction_mode == 'future':
            # For future prediction, we need features from the day before target_date
            feature_date = target_date - timedelta(days=1)
            print(
                f"📊 Using features from: {feature_date.strftime('%Y-%m-%d')} to predict {target_date.strftime('%Y-%m-%d')}")
        else:
            # For current prediction, use features from the same date
            feature_date = target_date
            print(
                f"📊 Using features from: {feature_date.strftime('%Y-%m-%d')} to predict current regime")

        # Check if feature_date exists in our data
        if feature_date not in self.df.index:
            # Find the closest available date
            available_dates = self.df.index
            closest_date = min(
                available_dates, key=lambda x: abs((x - feature_date).days))
            print(
                f"⚠️  Exact date not found. Using closest available date: {closest_date.strftime('%Y-%m-%d')}")
            feature_date = closest_date

        # Get feature columns
        if feature_set == 'scaled':
            feature_cols = [
                col for col in self.df.columns if col.endswith('_scaled')]
        elif feature_set == 'raw':
            feature_cols = ['close', 'high', 'low', 'open', 'volume',
                            'ma5', 'ma20', 'rsi', 'returns', 'volatility']
        elif feature_set == 'all':
            feature_cols = [col for col in self.df.columns if col != 'regime']

        # Filter existing columns and match training features
        feature_cols = [
            col for col in feature_cols if col in self.df.columns and col in self.feature_names]

        if not feature_cols:
            raise ValueError("No matching feature columns found!")

        # Extract features for the specific date
        try:
            X_input = self.df.loc[[feature_date], feature_cols]
        except KeyError:
            raise ValueError(f"Date {feature_date} not found in dataset!")

        # Handle missing values
        X_input = X_input.fillna(method='ffill').fillna(method='bfill')

        if X_input.isnull().any().any():
            print(
                "⚠️  Warning: Some features have missing values. Prediction may be unreliable.")

        # Make prediction
        prediction_encoded = self.best_model.predict(X_input)[0]
        predicted_regime = self.label_encoder.inverse_transform([prediction_encoded])[
            0]

        # Get prediction probabilities
        try:
            probabilities = self.best_model.predict_proba(X_input)[0]
            prob_dict = dict(zip(self.label_encoder.classes_, probabilities))
            max_prob = probabilities.max()
        except AttributeError:
            prob_dict = None
            max_prob = None
            print("⚠️  Probability predictions not available for this model")

        # Get actual regime if available (for validation)
        actual_regime = None
        if target_date in self.df.index and 'regime' in self.df.columns:
            actual_regime = self.df.loc[target_date, 'regime']

        # Prepare results
        results = {
            'target_date': target_date,
            'feature_date': feature_date,
            'predicted_regime': predicted_regime,
            'prediction_confidence': max_prob,
            'all_probabilities': prob_dict,
            'actual_regime': actual_regime,
            'prediction_mode': self.prediction_mode,
            'features_used': feature_cols
        }

        # Print results
        print(f"\n🎯 PREDICTION RESULTS")
        print(f"📅 Target Date: {target_date.strftime('%Y-%m-%d')}")
        print(f"📊 Feature Date: {feature_date.strftime('%Y-%m-%d')}")
        print(f"🔮 Predicted Regime: {predicted_regime}")
        if max_prob:
            print(f"📈 Confidence: {max_prob:.4f}")
        if actual_regime:
            print(f"✅ Actual Regime: {actual_regime}")
            print(f"🎯 Correct: {predicted_regime == actual_regime}")

        if prob_dict:
            print(f"\n📊 All Regime Probabilities:")
            for regime, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"   {regime}: {prob:.4f}")

        return results

    def predict_next_n_days(self, start_date, n_days, feature_set='scaled'):
        """
        Predict regimes for the next N days starting from start_date.
        
        Args:
            start_date: Starting date for predictions
            n_days: Number of days to predict
            feature_set: Feature set to use
            
        Returns:
            DataFrame with predictions
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)

        print(
            f"🔮 Predicting regimes for {n_days} days starting from {start_date.strftime('%Y-%m-%d')}")

        predictions = []

        for i in range(n_days):
            current_date = start_date + timedelta(days=i)

            try:
                result = self.predict_regime_for_date(
                    current_date, feature_set)
                predictions.append({
                    'date': current_date,
                    'predicted_regime': result['predicted_regime'],
                    'confidence': result['prediction_confidence'],
                    'actual_regime': result['actual_regime']
                })
                print(
                    f"Day {i+1}: {current_date.strftime('%Y-%m-%d')} → {result['predicted_regime']}")
            except Exception as e:
                print(
                    f"❌ Error predicting for {current_date.strftime('%Y-%m-%d')}: {str(e)}")
                continue

        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        predictions_df.set_index('date', inplace=True)

        return predictions_df

    def analyze_grid_search_results(self, top_n=10):
        """Analyze and visualize grid search results."""
        if self.grid_search is None:
            raise ValueError("Grid search not performed yet!")

        print(f"🔍 Analyzing top {top_n} parameter combinations...")

        # Convert results to DataFrame
        results_df = pd.DataFrame(self.grid_search.cv_results_)

        # Sort by test score
        top_results = results_df.nlargest(top_n, 'mean_test_score')

        print("\n🏆 Top parameter combinations:")
        for i, (idx, row) in enumerate(top_results.iterrows(), 1):
            print(
                f"{i}. Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
            print(f"   Parameters: {row['params']}")
            print()

        # Plot validation curves for different kernels
        kernels = results_df['param_svm__kernel'].unique()

        plt.figure(figsize=(15, 5))

        for i, kernel in enumerate(kernels, 1):
            if pd.isna(kernel):
                continue

            plt.subplot(1, len(kernels), i)
            kernel_results = results_df[results_df['param_svm__kernel'] == kernel]

            plt.scatter(range(len(kernel_results)), kernel_results['mean_test_score'],
                        alpha=0.7, s=50)
            plt.title(f'{kernel.upper()} Kernel Performance')
            plt.xlabel('Parameter Combination')
            plt.ylabel('CV Score')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return top_results

    def plot_regime_timeline(self, X_test, y_test, y_pred, sample_dates=None):
        """Plot regime predictions over time."""
        plt.figure(figsize=(15, 8))

        if sample_dates is not None:
            x_axis = sample_dates
            plt.xlabel('Date')
        else:
            x_axis = range(len(y_test))
            plt.xlabel('Sample Index')

        # Convert encoded labels back to regime names
        true_regimes = self.label_encoder.inverse_transform(y_test)
        pred_regimes = self.label_encoder.inverse_transform(y_pred)

        plt.subplot(2, 1, 1)
        plt.scatter(x_axis, true_regimes, alpha=0.7,
                    c='blue', label='True Regime')
        plt.title('True Regimes Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.scatter(x_axis, pred_regimes, alpha=0.7,
                    c='red', label='Predicted Regime')
        plt.title('Predicted Regimes Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if sample_dates is not None:
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def feature_importance_analysis(self):
        """Analyze feature importance (for linear kernel)."""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")

        if self.best_model.named_steps['svm'].kernel != 'linear':
            print("⚠️  Feature importance analysis only available for linear kernel")
            return None

        # Get feature coefficients
        svm_model = self.best_model.named_steps['svm']
        coef = np.abs(svm_model.coef_).mean(axis=0)

        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': coef
        }).sort_values('importance', ascending=False)

        print("📊 Feature Importance (Linear Kernel):")
        print(feature_importance)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance.head(
            15), x='importance', y='feature')
        plt.title('Top 15 Feature Importance - Regime Detection')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

        return feature_importance


def main():
    """Main execution function with comprehensive workflow."""

    # 🔥 Initialize detector with FUTURE prediction mode
    detector = SVMRegimeDetector(
        cv_folds=5,
        test_size=0.2,
        random_state=42,
        prediction_mode='future'  # This is the key change!
    )

    # Load data (replace with your CSV path)
    csv_path = "ticker/AAPL/AAPL_data.csv"  # Update this path
    try:
        df = detector.load_data(csv_path)
    except FileNotFoundError:
        print("❌ CSV file not found. Please update the csv_path variable.")
        print("💡 Expected format: Date,close,high,low,open,volume,ma5,ma20,rsi,returns,volatility,regime,open_scaled,high_scaled,low_scaled,close_scaled,volume_scaled,ma5_scaled,ma20_scaled,rsi_scaled,volatility_scaled")
        return

    # Prepare features (using scaled features for better SVM performance)
    X, y = detector.prepare_features(feature_set='scaled')

    # Split data (temporal split for time series)
    X_train, X_test, y_train, y_test = detector.split_data(
        X, y, method='temporal')

    # Train model with comprehensive grid search
    print("\n" + "="*60)
    print("🤖 TRAINING SVM WITH GRID SEARCH (FUTURE PREDICTION MODE)")
    print("="*60)

    best_model = detector.train_model(
        X_train, y_train,
        kernel_types=['rbf', 'linear', 'poly'],
        cv_method='timeseries',
        scoring='accuracy',
        n_jobs=-1
    )

    # Evaluate model
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)

    results = detector.evaluate_model(X_test, y_test)

    # 🎯 PREDICT FOR SPECIFIC DATES
    print("\n" + "="*60)
    print("🔮 PREDICTING FOR SPECIFIC DATES")
    print("="*60)

    # Example 1: Predict for a specific date
    target_date = "2023-12-15"  # Change this to your desired date
    prediction_result = detector.predict_regime_for_date(target_date)

    # Example 2: Predict for multiple dates
    print("\n" + "-"*40)
    print("📅 MULTIPLE DATE PREDICTIONS")
    print("-"*40)

    start_date = "2025-07-01"
    predictions_df = detector.predict_next_n_days(start_date, n_days=10)
    print(f"\n📊 Predictions Summary:")
    print(predictions_df)

    # Analyze grid search results
    print("\n" + "="*60)
    print("🔍 GRID SEARCH ANALYSIS")
    print("="*60)

    top_results = detector.analyze_grid_search_results(top_n=5)

    print("\n" + "="*60)
    print("✅ REGIME DETECTION ANALYSIS COMPLETE")
    print("="*60)
    print(f"🎯 Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"🏆 Best Model: {detector.grid_search.best_params_}")
    print(f"🔮 Prediction Mode: {detector.prediction_mode}")


if __name__ == "__main__":
    main()
