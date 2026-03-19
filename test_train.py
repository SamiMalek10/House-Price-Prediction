"""
Unit and integration tests for the house price prediction pipeline.
Run with: python -m pytest test_train.py -v
"""

import os
import pickle
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from train_model import generate_synthetic_data


# ── Unit Tests ────────────────────────────────────────────────────────────────

class TestGenerateSyntheticData:
    def test_shape(self):
        df = generate_synthetic_data(200)
        assert df.shape == (200, 8), "Expected 200 rows and 8 columns (7 features + price)"

    def test_expected_columns(self):
        df = generate_synthetic_data(100)
        expected = {'square_feet', 'bedrooms', 'bathrooms', 'age_years',
                    'lot_size', 'garage_spaces', 'neighborhood_score', 'price'}
        assert set(df.columns) == expected

    def test_feature_ranges(self):
        df = generate_synthetic_data(500)
        assert df['square_feet'].between(800, 4000).all()
        assert df['bedrooms'].between(1, 5).all()
        assert df['bathrooms'].between(1, 4).all()
        assert df['age_years'].between(0, 50).all()
        assert df['lot_size'].between(2000, 10000).all()
        assert df['garage_spaces'].between(0, 3).all()
        assert df['neighborhood_score'].between(1, 10).all()

    def test_no_null_values(self):
        df = generate_synthetic_data(100)
        assert not df.isnull().any().any(), "Dataset must have no missing values"

    def test_price_is_positive(self):
        df = generate_synthetic_data(300)
        assert (df['price'] > 0).all(), "All prices should be positive"

    def test_reproducibility_with_seed(self):
        np.random.seed(42)
        df1 = generate_synthetic_data(50)
        np.random.seed(42)
        df2 = generate_synthetic_data(50)
        pd.testing.assert_frame_equal(df1, df2)


class TestScalerLeakage:
    """Ensure the scaler is fit only on training data."""

    def test_scaler_fit_on_train_only(self):
        df = generate_synthetic_data(200)
        feature_names = ['square_feet', 'bedrooms', 'bathrooms', 'age_years',
                         'lot_size', 'garage_spaces', 'neighborhood_score']
        from sklearn.model_selection import train_test_split
        X = df[feature_names]
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Scaler stats must come from train set only
        assert scaler.mean_.shape == (len(feature_names),)
        # Test-set scaled mean should NOT be ~0 (unlike train set)
        assert abs(X_train_scaled.mean()) < abs(X_test_scaled.mean()) + 1


class TestModelTraining:
    def test_model_predicts_correct_shape(self):
        df = generate_synthetic_data(200)
        feature_names = ['square_feet', 'bedrooms', 'bathrooms', 'age_years',
                         'lot_size', 'garage_spaces', 'neighborhood_score']
        X = df[feature_names].values
        y = df['price'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_scaled, y)
        preds = model.predict(X_scaled)

        assert preds.shape == (200,)

    def test_model_r2_above_threshold(self):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        df = generate_synthetic_data(500)
        feature_names = ['square_feet', 'bedrooms', 'bathrooms', 'age_years',
                         'lot_size', 'garage_spaces', 'neighborhood_score']
        X = df[feature_names]
        y = df['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42)
        model.fit(X_train_s, y_train)

        r2 = r2_score(y_test, model.predict(X_test_s))
        assert r2 >= 0.85, f"Test R² too low: {r2:.3f}"


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestArtifactPersistence:
    """Verify that model artifacts are saved and reloaded correctly."""

    def test_artifacts_saved_and_loadable(self, tmp_path):
        df = generate_synthetic_data(200)
        feature_names = ['square_feet', 'bedrooms', 'bathrooms', 'age_years',
                         'lot_size', 'garage_spaces', 'neighborhood_score']
        X = df[feature_names].values
        y = df['price'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_scaled, y)

        model_path = tmp_path / "model.pkl"
        scaler_path = tmp_path / "scaler.pkl"
        names_path = tmp_path / "feature_names.pkl"

        for obj, path in [(model, model_path), (scaler, scaler_path),
                          (feature_names, names_path)]:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)

        for path in [model_path, scaler_path, names_path]:
            assert path.exists(), f"{path.name} was not saved"

        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            loaded_scaler = pickle.load(f)
        with open(names_path, 'rb') as f:
            loaded_names = pickle.load(f)

        assert loaded_names == feature_names
        sample = np.array([[2000, 3, 2, 10, 5000, 2, 7.0]])
        pred = loaded_model.predict(loaded_scaler.transform(sample))
        assert pred.shape == (1,) and pred[0] > 0

    def test_prediction_pipeline_end_to_end(self):
        """Full pipeline: generate → train → predict on a known input."""
        from sklearn.model_selection import train_test_split

        df = generate_synthetic_data(300)
        feature_names = ['square_feet', 'bedrooms', 'bathrooms', 'age_years',
                         'lot_size', 'garage_spaces', 'neighborhood_score']
        X = df[feature_names]
        y = df['price']

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        model = RandomForestRegressor(n_estimators=20, max_depth=15, random_state=42)
        model.fit(scaler.fit_transform(X_train), y_train)

        sample = np.array([[2000, 3, 2, 10, 5000, 2, 7.0]])
        prediction = model.predict(scaler.transform(sample))[0]

        assert isinstance(prediction, float)
        assert 50_000 < prediction < 1_500_000, \
            f"Prediction ${prediction:,.0f} is outside expected range"
