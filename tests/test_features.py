import pytest

from prodml.features import prepare_features


@pytest.mark.parametrize(
    "input_data, expected_pu_do, expected_distance",
    [
        (
            {"PULocationID": "43", "DOLocationID": "236", "trip_distance": 2.5},
            "43_236",
            2.5,
        ),
        ({"DOLocationID": "236", "trip_distance": 2.5}, "Unknown_236", 2.5),
        (
            {"PULocationID": "43", "DOLocationID": "236", "trip_distance": 0.0},
            "43_236",
            0.0,
        ),
        (
            {"PULocationID": "999", "DOLocationID": "999", "trip_distance": 5.0},
            "999_999",
            5.0,
        ),
    ],
)
def test_feature_engineering_edge_cases(input_data, expected_pu_do, expected_distance):
    features = prepare_features(input_data)
    assert features["PU_DO"] == expected_pu_do
    assert features["trip_distance"] == expected_distance
