from app.feature_adapters.informer_adapter import InformerFeatureAdapter


def create_adapter(config):
    return InformerFeatureAdapter(config)
