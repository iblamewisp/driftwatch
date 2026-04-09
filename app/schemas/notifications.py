from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, computed_field


class DriftAlertPayload(BaseModel):
    detected_at: datetime
    cluster_id: UUID | None
    baseline_score: float
    current_score: float
    delta_percent: float
    threshold_percent: float
    alert_channel: str

    @computed_field
    @property
    def summary(self) -> str:
        cluster = f"cluster {self.cluster_id}" if self.cluster_id else "unknown cluster"
        return (
            f"Drift detected in {cluster}: {self.delta_percent:.1f}% drop "
            f"(baseline {self.baseline_score:.3f} → current {self.current_score:.3f})"
        )
