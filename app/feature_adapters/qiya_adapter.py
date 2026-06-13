from app.feature_adapters.xgb_adapter import XGBFeatureAdapter


def create_adapter(config):
    return XGBFeatureAdapter("qiya", config["active_model_dir"], tau_minutes=10)
