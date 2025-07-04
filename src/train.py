# dependencies
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, models=None, param_grids=None, random_state=42):
        """
        Initialize ModelTrainer with models and hyperparameter grids.

        Parameters:
        - models: Dict of model names and model instances
        - param_grids: Dict of model names and hyperparameter grids
        - random_state: Random seed for reproducibility
        """
        self.models = models or {
            'RandomForest': RandomForestClassifier(random_state=random_state),
            'GradientBoosting': GradientBoostingClassifier(
                random_state=random_state)
        }
        self.param_grids = param_grids or {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }
        self.random_state = random_state
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None

    def split_data(self, df, target_col, test_size=0.3):
        """
        Split data into training and testing sets,

        Parameters:
        - df: Input DataFrame
        - target_col: Target column name
        - test_size: Proportion of test set

        Returns:
        - X_train, X_test, y_train, y_test
        """
        # Exclude RFM-related columns to prevent data leakage
        exclude_cols = [target_col, 'Recency',
                        'Frequency', 'Monetary', 'Cluster']
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=self.random_state, stratify=y
        )

        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=self.random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, y_true, y_pred, y_pred_proba):
        """
        Evaluate model performance using multiple metrics.

        Parameters:
        - y_true: True labels
        - y_pred: Predicted labels
        - y_pred_proba: Predicted probabilities for positive class

        Returns:
        - Dictionary of evaluation metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }

    def train_and_tune(self, X_train, y_train, X_test, y_test,
                       experiment_name="CreditRiskExperiment"):
        """
        Train models with hyperparameter tuning and log to MLflow.

        Parameters:
        - X_train, y_train: Training data
        - X_test, y_test: Testing data
        - experiment_name: MLflow experiment name

        Returns:
        - Dictionary of model results
        """
        mlflow.set_experiment(experiment_name)
        results = {}

        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                # Log parameters
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("random_state", self.random_state)

                # Hyperparameter tuning
                grid_search = GridSearchCV(
                    model,
                    self.param_grids[model_name],
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)

                # Get best model
                best_model = grid_search.best_estimator_
                mlflow.log_params(grid_search.best_params_)

                # Predictions
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]

                # Evaluate
                metrics = self.evaluate_model(y_test, y_pred, y_pred_proba)
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log model with input example
                input_example = X_train.head(1)
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path=model_name,
                    registered_model_name=model_name,
                    input_example=input_example
                )

                # Update best model
                if metrics['f1'] > self.best_score:
                    self.best_score = metrics['f1']
                    self.best_model = best_model
                    self.best_model_name = model_name

                results[model_name] = {
                    'model': best_model,
                    'metrics': metrics,
                    'best_params': grid_search.best_params_
                }

                print(f"{model_name} Metrics: {metrics}")
                print(f"{model_name} Best Params: {grid_search.best_params_}")

        return results

    def register_best_model(self, model_name, model, run_id):
        """
        Register the best model in MLflow Model Registry.

        Parameters:
        - model_name: Name of the model
        - model: Trained model instance
        - run_id: MLflow run ID
        """
        model_uri = f"runs:/{run_id}/{model_name}"
        mlflow.register_model(model_uri, model_name)
        print(f"Registered {model_name} in MLflow Model Registry")


# Example usage
if __name__ == "__main__":
    pass
