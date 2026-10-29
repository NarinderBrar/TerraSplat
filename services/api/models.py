"""Public contracts for the read-only viewer API."""

from typing import Literal

from pydantic import BaseModel, Field


class CaptureDates(BaseModel):
    train: list[str]
    holdout: list[str]


class Anchor(BaseModel):
    latitude: float
    longitude: float
    ellipsoidal_height_m: float


class Polygon(BaseModel):
    type: Literal['Polygon'] = 'Polygon'
    coordinates: list[list[list[float]]]


class Quality(BaseModel):
    registered_lidar_mae_all_m: float
    registered_lidar_mae_ground_m: float
    registered_lidar_mae_buildings_m: float
    registered_lidar_mae_vegetation_m: float
    registered_lidar_rmse_m: float
    holdout_psnr_db: float
    horizontal_offset_m: float
    completeness: float
    splats: int
    tiles: int
    glb_bytes: int


class Source(BaseModel):
    run_directory: str
    tiles_directory: str
    config_sha256: str
    tile_plan_sha256: str
    tiles_report_sha256: str
    export_manifest_sha256: str
    fused_ply_sha256: str


class Manifest(BaseModel):
    schema_version: Literal[1]
    dataset_id: str
    release_id: str
    title: str
    description: str
    created_at: str
    pipeline_git_commit: str
    source: Source
    capture_dates: CaptureDates
    historical: Literal[True]
    anchor: Anchor
    footprint: Polygon
    bbox: list[float] = Field(min_length=6, max_length=6)
    tileset_path: str
    tile_count: int
    bytes: int
    quality: Quality
    evidence: list[str]
    files: dict[str, str]
    limitations: str


class ReleaseSummary(BaseModel):
    release_id: str
    title: str
    manifest_url: str
    tileset_url: str
    capture_dates: CaptureDates
    quality: Quality
    bytes: int


class DatasetSummary(BaseModel):
    dataset_id: str
    title: str
    region: str
    active_release_id: str | None
    active_release: ReleaseSummary | None


class Dataset(DatasetSummary):
    releases: list[ReleaseSummary]


class EvidenceLink(BaseModel):
    name: str
    url: str
    sha256: str


class QualityReport(BaseModel):
    release_id: str
    quality: Quality
    evidence: list[EvidenceLink]
    limitations: str


class FootprintProperties(BaseModel):
    dataset_id: str
    release_id: str
    title: str
    capture_date_start: str
    capture_date_end: str
    headline_mae_m: float
    tileset_url: str


class Feature(BaseModel):
    type: Literal['Feature'] = 'Feature'
    geometry: Polygon
    properties: FootprintProperties


class FeatureCollection(BaseModel):
    type: Literal['FeatureCollection'] = 'FeatureCollection'
    features: list[Feature]


class IndexLink(BaseModel):
    region: str
    tileset_url: str


class Health(BaseModel):
    status: Literal['ok'] = 'ok'
