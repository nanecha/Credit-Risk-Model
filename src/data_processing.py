import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xverse.transformer import WOE
import warnings
import numpy as np
warnings.filterwarnings('ignore')
# This code defines a feature engineering pipelines


class WOEWrapper:
    """
    Wrapper for xverse.WOE to handle NumPy array inputs by converting to
    DataFrame
    """

    def __init__(self, woe_transformer=WOE()):
        self.woe_transformer = woe_transformer
        self.feature_names = None

    def fit(self, X, y=None):
        # Convert NumPy array to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        self.woe_transformer.fit(X, y)
        self.feature_names = X.columns
        return self

    def transform(self, X):
        # Convert NumPy array to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.woe_transformer.transform(X)

    def fit_transform(self, X, y=None):
        # Convert NumPy array to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.woe_transformer.fit_transform(X, y)

    @property
    def iv_df(self):
        return self.woe_transformer.iv_df

    @property
    def woe_df(self):
        return self.woe_transformer.woe_df


class FeatureEngineeringPipeline:
    def __init__(self):
        self.pipeline = None
        self.categorical_columns = []
        self.numerical_columns = []

    def extract_datetime_features(self, df, datetime_column):
        """Extract datetime features from a timestamp column"""
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df['transaction_hour'] = df[datetime_column].dt.hour
        df['transaction_day'] = df[datetime_column].dt.day
        df['transaction_month'] = df[datetime_column].dt.month
        df['transaction_year'] = df[datetime_column].dt.year
        return df

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

    def calculate_woe_iv(self, df, categorical_columns, target_col):
        """Calculate WOE and IV for categorical variables"""
        woe_transformer = WOE()
        woe_transformer.fit(df[categorical_columns], df[target_col])

        # Get WOE and IV values
        iv_summary = woe_transformer.iv_df
        woe_dict = woe_transformer.woe_df

        # Ensure IV summary is a DataFrame with Variable_Name and
        # Information_Value
        if iv_summary is None or iv_summary.empty:
            iv_summary = pd.DataFrame({
                'Variable_Name': categorical_columns,
                'Information_Value': [0] * len(categorical_columns)
            })

        return iv_summary, woe_dict

    def fit_transform(self, df, target_col=None, datetime_col=None,
                      group_by_col=None, amount_col=None):
        """Fit and transform the data through the feature
        #engineering pipeline"""
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

        # Calculate WOE and IV for categorical columns if target is provided
        iv_summary = None
        woe_dict = None
        if target_col and self.categorical_columns:
            iv_summary, woe_dict = self.calculate_woe_iv(
                df_processed, self.categorical_columns, target_col)

        # Define preprocessing steps
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant',
                                      fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore',
                                     sparse_output=False))
        ])

        # Use WOEWrapper for categorical variables if target is provided
        if target_col:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant',
                                          fill_value='missing')),
                ('woe', WOEWrapper())
            ])

        # Combine preprocessing steps
        preprocessor_steps = [
            ('num', numerical_transformer, self.numerical_columns),
            ('cat', categorical_transformer, self.categorical_columns)
        ]

        # Create the main pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(
                transformers=preprocessor_steps))
        ])

        # Fit and transform the data
        transformed_data = self.pipeline.fit_transform(df_processed)

        # Generate feature names
        feature_names = []
        for name, transformer, columns in \
                self.pipeline.named_steps['preprocessor'].transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                try:
                    # For OneHotEncoder
                    feature_names.extend(
                        transformer.named_steps
                        ['onehot'].get_feature_names_out(
                            columns
                        )
                    )
                except Exception:
                    # For WOEWrapper or other cases
                    feature_names.extend(columns)

        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)

        # Add target column back if it exists
        if target_col:
            transformed_df[target_col] = df_processed[target_col].values

        return transformed_df, iv_summary, woe_dict

    def transform(self, df, datetime_col=None, group_by_col=None,
                  amount_col=None):
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
        for name, transformer, columns in \
                self.pipeline.named_steps['preprocessor'].transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                try:
                    feature_names.extend(
                        transformer.named_steps['onehot']
                        .get_feature_names_out(columns)
                    )
                except Exception:
                    feature_names.extend(columns)  # For WOE case

        return pd.DataFrame(transformed_data, columns=feature_names)


# Example usage
if __name__ == "__main__":
    pass
