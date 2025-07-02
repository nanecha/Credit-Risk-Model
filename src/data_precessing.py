import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xverse.transformer import WOE
import warnings
warnings.filterwarnings('ignore')
# This code defines a feature engineering pipelines


class FeatureEngineeringPipeline:
    def __init__(self):
        self.pipeline = None
        self.label_encoders = {}
        self.categorical_columns = []
        self.numerical_columns = []

    # Create aggregate features for transactions

    def create_aggregate_features(self, df, group_by_col, amount_col):
        """Create aggregate features for transactions"""
        agg_features = df.groupby(group_by_col)[amount_col].agg([
            ('total_amount', 'sum'),
            ('avg_amount', 'mean'),
            ('trans_count', 'count'),
            ('std_amount', 'std')
        ]).reset_index()

        # Fill NaN in std_amount with 0 (for cases with single transaction)
        agg_features['std_amount'] = agg_features['std_amount'].fillna(0)

        return df.merge(agg_features, on=group_by_col, how='left')

    # Extract datetime features from a TransactionStartTime column

    def extract_datetime_features(self, df, datetime_column):
        """Extract datetime features from a timestamp column"""
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df['transaction_hour'] = df[datetime_column].dt.hour
        df['transaction_day'] = df[datetime_column].dt.day
        df['transaction_month'] = df[datetime_column].dt.month
        df['transaction_year'] = df[datetime_column].dt.year
        return df

    def fit_transform(self, df, target_col=None, datetime_col=None,
                      group_by_col=None, amount_col=None):
        """
        Fit and transform the data through the feature engineering pipeline
        """
        # Make a copy to avoid modifying the original dataframe
        df_processed = df.copy()

        # Extract datetime features if datetime column is provided
        if datetime_col:
            df_processed = self.extract_datetime_features(
                df_processed, datetime_col)

        # Create aggregate features if group_by and amount columns are provided
        if group_by_col and amount_col:
            df_processed = self.create_aggregate_features(
                df_processed, group_by_col, amount_col)

        # Identify column types
        self.categorical_columns = df_processed.select_dtypes(
            include=['object', 'category']).columns
        self.numerical_columns = df_processed.select_dtypes(
            include=['int64', 'float64']).columns

        # Remove target column from feature processing if provided
        if target_col:
            self.categorical_columns = [
                col for col in self.categorical_columns if col != target_col]
            self.numerical_columns = [
                col for col in self.numerical_columns if col != target_col]

        # Define preprocessing steps
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(
                strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ))
        ])

        # Create WOE transformer for categorical variables
        # if target is provided
        if target_col:
            woe_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(
                    strategy='constant', fill_value='missing')),
                ('woe', WOE())
            ])

        # Combine preprocessing steps
        preprocessor_steps = [
            ('num', numerical_transformer, self.numerical_columns),
            (
                'cat',
                categorical_transformer if not target_col else woe_transformer,
                self.categorical_columns
            )
        ]

        # Create the main pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor',
             ColumnTransformer(transformers=preprocessor_steps)
             )
        ])

        # Fit and transform the data
        transformed_data = self.pipeline.fit_transform(
            df_processed.drop(columns=[target_col] if target_col else []),
            df_processed[target_col] if target_col else None
        )

        # Get feature names after transformation
        feature_names = []
        preprocessor = self.pipeline.named_steps['preprocessor']
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat' and not target_col:
                feature_names.extend(
                    transformer.named_steps['onehot']
                    .get_feature_names_out(columns)
                )
            else:
                # WOE keeps original column names
                feature_names.extend(columns)

        # Convert to DataFrame
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)

        # Add target column back if it exists
        if target_col:
            transformed_df[target_col] = df_processed[target_col].values

        return transformed_df

    def transform(
        self,
        df,
        datetime_col=None,
        group_by_col=None,
        amount_col=None
    ):
        """Transform new data using the fitted pipeline"""
        df_processed = df.copy()

        if datetime_col:
            df_processed = self.extract_datetime_features(
                df_processed, datetime_col)

        if group_by_col and amount_col:
            df_processed = self.create_aggregate_features(
                df_processed, group_by_col, amount_col)

        transformed_data = self.pipeline.transform(df_processed)

        feature_names = []
        preprocessor = self.pipeline.named_steps['preprocessor']
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                try:
                    feature_names.extend(
                        transformer.named_steps['onehot']
                        .get_feature_names_out(columns)
                    )
                except AttributeError:
                    feature_names.extend(columns)  # For WOE case

        return pd.DataFrame(transformed_data, columns=feature_names)


# Example usage
if __name__ == "__main__":
    pass  # Add your example usage or tests here