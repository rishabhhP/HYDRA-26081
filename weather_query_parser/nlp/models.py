"""Pydantic schemas for the HYDRA weather-query parser."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


WeatherVariable = Literal["rainfall", "temperature", "wind", "humidity", "thunderstorm", "wind_gust", "heat_stress", "soil_moisture", "radar", "satellite"]
ExtremeEventType = Literal["heavy_rain", "heatwave", "high_wind"]
LeadHours = Literal[0, 24, 48, 72]
TimeRangeKind = Literal["lead_hours", "date_range", "ambiguous", "unspecified"]
QueryScope = Literal["selected_location", "spatial_ranking"]
RankingScope = Literal["locations", "states"]
DataDomain = Literal["hydra_model", "live_layers", "imagery", "historical_observations"]
UseCase = Literal["general", "agriculture", "aviation", "marine", "energy", "urban", "emergency"]
QuestionOperation = Literal["location_ranking", "time_ranking", "aggregate", "point_lookup", "comparison", "explanation", "capability"]
RankingDirection = Literal["ascending", "descending"]
ExplanationTopic = Literal["architecture", "weights", "uncertainty", "events", "data_coverage", "provenance", "forecast_value"]


class ParseRequest(BaseModel):
    """Request body accepted by the standalone ``/parse`` endpoint."""

    query: str = Field(min_length=1, max_length=2000)


class Coordinates(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class TimeRange(BaseModel):
    """A HYDRA-compatible point lead or an explicit interval when needed."""

    kind: TimeRangeKind
    lead_hours: LeadHours | None = None
    start_date: str | None = None
    end_date: str | None = None
    text: str | None = None


class ParsedQuery(BaseModel):
    location: str | None = None
    location_suggestion: str | None = None
    location_candidate: str | None = None
    coordinates: Coordinates | None = None
    weather_variable: WeatherVariable | None = None
    # A request such as "weather today in Kerala" asks for a concise set of
    # current conditions rather than a single model target.
    weather_summary_request: bool = False
    data_layer: str | None = None
    data_domain: DataDomain = "hydra_model"
    use_case: UseCase = "general"
    query_bucket: str | None = None
    question_operation: QuestionOperation = "point_lookup"
    ranking_direction: RankingDirection = "descending"
    explanation_topic: ExplanationTopic = "forecast_value"
    model_evidence_request: bool = False
    analysis_window_days: int | None = None
    analysis_window_text: str | None = None
    time_range: TimeRange
    extreme_event_type: ExtremeEventType | None = None
    query_scope: QueryScope = "selected_location"
    # State rankings use a disclosed representative coordinate per state/UT.
    # They are never silently treated as statewide spatial averages.
    ranking_scope: RankingScope = "locations"
    grid_cell: str | None = None

    def as_response(self) -> dict:
        return self.model_dump() if hasattr(self, "model_dump") else self.dict()
