import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SVMRegimeDetector:
    """
    SVM-based regime detection system for financial markets.
    Supports multiple kernels and hyperparameter optimization via Grid Search.
    """

    def __init__(self, cv_folds=5, test_size=0.2, random_state=42):
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.grid_search = None
        self.feature_names = None

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
        print(f"🔧 Preparing features (set: {feature_set})...")

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

        print(
            f"🔍 Grid search will test {sum(len(p['svm__C']) * len(p.get('svm__gamma', [1])) * len(p.get('svm__degree', [1])) * len(p['svm__class_weight']) for p in param_grid)} combinations")

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

    def predict_regime(self, X_new):
        """Predict regime for new data."""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")

        predictions = self.best_model.predict(X_new)

        # Only get probabilities if available
        try:
            probabilities = self.best_model.predict_proba(X_new)
        except AttributeError:
            probabilities = None
            print("⚠️  Probability predictions not available for this model")

        # Convert back to regime names
        regime_predictions = self.label_encoder.inverse_transform(predictions)

        return regime_predictions, probabilities

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

# Example usage and main execution


def main():
    """Main execution function with comprehensive workflow."""

    # Initialize detector
    detector = SVMRegimeDetector(cv_folds=5, test_size=0.2, random_state=42)

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
    print("🤖 TRAINING SVM WITH GRID SEARCH")
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

    # Analyze grid search results
    print("\n" + "="*60)
    print("🔍 GRID SEARCH ANALYSIS")
    print("="*60)

    top_results = detector.analyze_grid_search_results(top_n=5)

    # Feature importance (if linear kernel)
    if detector.best_model.named_steps['svm'].kernel == 'linear':
        print("\n" + "="*60)
        print("📈 FEATURE IMPORTANCE")
        print("="*60)
        detector.feature_importance_analysis()

    # Plot regime timeline
    sample_dates = df.index[-len(y_test):] if hasattr(df.index,
                                                      'to_pydatetime') else None
    detector.plot_regime_timeline(
        X_test, y_test, results['predictions'], sample_dates)

    print("\n" + "="*60)
    print("✅ REGIME DETECTION ANALYSIS COMPLETE")
    print("="*60)
    print(f"🎯 Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"🏆 Best Model: {detector.grid_search.best_params_}")


if __name__ == "__main__":
    main()
