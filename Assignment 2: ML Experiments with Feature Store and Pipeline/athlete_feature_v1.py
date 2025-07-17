from feast import Entity, FeatureView, Field, FileSource
from feast import types
from feast.value_type import ValueType
from datetime import timedelta

# Define entity
athlete = Entity(
    name="athlete_id",
    value_type=ValueType.INT64,
    description="Athlete identifier"
)

# Define source pointing to our actual data
athletes_source_v1 = FileSource(
    path="data/feature_v1.csv",
    timestamp_field="event_timestamp",
)

# Define FeatureView V1 - Basic physical features
athlete_features_v1 = FeatureView(
    name="athlete_features_v1",
    entities=[athlete],
    ttl=timedelta(days=30),
    schema=[
        Field(name="age", dtype=types.Float32),
        Field(name="height", dtype=types.Float32),
        Field(name="weight", dtype=types.Float32),
        Field(name="deadlift", dtype=types.Float32),
        Field(name="backsq", dtype=types.Float32),
        Field(name="pullups", dtype=types.Float32),
        Field(name="is_high_performer", dtype=types.Int32),
    ],
    source=athletes_source_v1
)