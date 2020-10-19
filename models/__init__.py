from .betavae import BetaVAE
from .gvae import GroupVAE


MODELS = {
    "GroupVAE": GroupVAE,
    "BetaVAE": BetaVAE
}


def get_model(config):
    model_name = config["models"]["name"]
    if MODELS.get(model_name) is None:
        raise NotImplementedError

    model = MODELS[model_name]
    return model(**config["models"]["params"])
