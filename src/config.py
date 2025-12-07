from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.parent.resolve()


class PathsConfig(BaseModel):
    data_raw: Path = BASE_DIR / "data" / "raw"
    data_processed: Path = BASE_DIR / "data" / "processed"
    submissions: Path = BASE_DIR / "data" / "submissions"


class MLflowConfig(BaseModel):
    uri: str = "databricks"
    experiment_name: str = "/Shared/Fraud_detection"
    catalog: str = "workspace"
    d_brick_schema: str = "default"


class XGBParams(BaseModel):
    n_estimators: int = 5000
    learning_rate: float = 0.01
    max_depth: int = 10
    tree_method: str = "hist"
    device: str = "cuda"  # Tu 4080
    early_stopping: int = 100


class LGBMParams(BaseModel):
    n_estimators: int = 5000
    learning_rate: float = 0.01
    device: str = "gpu"


class Settings(BaseSettings):
    project_name: str = "IEEE_Fraud_Detection"
    seed: int = 42

    paths: PathsConfig = PathsConfig()
    mlflow: MLflowConfig = MLflowConfig()

    xgboost: XGBParams = XGBParams()
    lightgbm: LGBMParams = LGBMParams()

    drop_cols: list[str] = ["isFraud", "TransactionID"]
    target: str = "isFraud"


settings = Settings()

