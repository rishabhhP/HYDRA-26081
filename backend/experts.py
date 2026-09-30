"""Extensible contract for future forecast sources; existing science stays upstream."""
from typing import Protocol, Any


class ForecastExpert(Protocol):
    name: str
    version: str

    def metadata(self) -> dict[str, Any]: ...
    def health(self) -> dict[str, Any]: ...
    def infer(self, history: Any, issue_date: str, leads: list[int], cells: list[int] | None = None) -> Any: ...
    def uncertainty(self) -> dict[str, Any]: ...


class NwpblendExpert:
    name = 'HYDRA adaptive blend'
    version = 'blocked'

    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.model = None

    def metadata(self):
        import json
        return json.loads((self.artifacts/'manifest.json').read_text())

    def health(self):
        return {'loaded': self.model is not None,
                'shared_statistics_available': (self.artifacts/'climatology_by_doy.npz').exists()}

    def infer(self, history, issue_date, leads, cells=None):
        from nwpblend.inference import BlendingForecaster
        if self.model is None:
            self.model = BlendingForecaster(self.artifacts)
        return self.model.forecast(history,issue_date,leads=leads,cells=cells)

    def uncertainty(self):
        return {'method': 'LightGBM quantiles and season-conditioned conformal intervals',
                'confidence_score': None, 'disagreement': 'standard deviation across expert values'}
