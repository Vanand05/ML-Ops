from feast import Entity, FeatureView, Field, FileSource
from feast import types
from feast.value_type import ValueType
from datetime import timedelta

# Define entity (same as V1)
athlete = Entity(
    name="athlete_id",
    value_type=ValueType.INT64,
    description="Athlete identifier"
)

# Define source pointing to our V2 data
athletes_source_v2 = FileSource(
    path="data/feature_v2.csv",
    timestamp_field="event_timestamp",
)

# Define FeatureView V2 - Performance-based features
athlete_features_v2 = FeatureView(
    name="athlete_features_v2",
    entities=[athlete],
    ttl=timedelta(days=365),
    schema=[
        Field(name="fran", dtype=types.Float32),
        Field(name="helen", dtype=types.Float32),
        Field(name="grace", dtype=types.Float32),
        Field(name="run400", dtype=types.Float32),
        Field(name="run5k", dtype=types.Float32),
        Field(name="snatch", dtype=types.Float32),
        Field(name="is_high_performer", dtype=types.Int32),
    ],
    source=athletes_source_v2
)