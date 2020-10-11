import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from torch.jit import trace


def export_model(model, path, input_shape=(1, 3, 64, 64), use_script_module=True):
    """
    Exports the model. If the model is a `ScriptModule`, it is saved as is. If not,
    it is traced (with the given input_shape) and the resulting ScriptModule is saved
    (this requires the `input_shape`, which defaults to the competition default).
    Parameters
    ----------
    model : torch.nn.Module or torch.jit.ScriptModule
        Pytorch Module or a ScriptModule.
    path : str
        Path to the file where the model is saved. Defaults to the value set by the
        `get_model_path` function above.
    input_shape : tuple or list
        Shape of the input to trace the module with. This is only required if model is not a
        torch.jit.ScriptModule.
    use_script_module : True or False (default = True)
        If True saves model as torch.jit.ScriptModule -- this is highly recommended.
        Setting it to False may cause later evaluation to fail.
    Returns
    -------
    str
        Path to where the model is saved.
    """
    input_shape = [1] + input_shape

    model = deepcopy(model).cpu().eval()
    if isinstance(model, torch.jit.ScriptModule):
        assert use_script_module, "Provided model is a ScriptModule, please set use_script_module to True."
    if use_script_module:
        if not isinstance(model, torch.jit.ScriptModule):
            assert input_shape is not None, "`input_shape` must be provided since model is not a " \
                                            "`ScriptModule`."
            traced_model = trace(model, torch.zeros(*input_shape))
        else:
            traced_model = model
        torch.jit.save(traced_model, str(path))
    else:
        torch.save(model, str(path))  # saves model as a nn.Module
    return path


def import_model(path):
    """
    Imports a model (as torch.jit.ScriptModule or torch.nn.Module) from file.
    By default the file is imported as torch.jit.ScriptModule. If it fails due to saved model being torch.nn.Module, the file is imported as torch.nn.Module.
    Parameters
    ----------
    path : str
        Path to where the model is saved. Defaults to the return value of the `get_model_path`
    Returns
    -------
    torch.jit.ScriptModule / torch.nn.Module
        The model file.
    """

    try:
        return torch.jit.load(str(path))
    except RuntimeError:
        try:
            return torch.load(str(path))  # loads model as a nn.Module
        except Exception as e:
            raise IOError("Could not load file. Please save as torch.jit.ScriptModule instead.") from e


class RepresentationExtractor(nn.Module):
    VALID_MODES = ["mean", "sample"]

    def __init__(self, encoder, mode="mean"):
        super().__init__()
        assert mode in self.VALID_MODES, f"`mode` mus be one of {self.VALID_MODES}"

        self.encoder = encoder
        self.mode = mode

    def forward(self, x):
        mu, logvar = self.encoder(x)
        if self.mode == "mean":
            return mu
        elif self.mode == "sample":
            return self.reparametrize(mu, logvar)
        else:
            raise NotImplementedError

    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


def make_representor(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def _represent(x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray), \
            "Input to the representation function must be a ndarray"
        assert x.ndim == 4, \
            "Input to the representation function must be a four dimensional NHWC tensor"

        x = np.moveaxis(x, 3, 1)
        x = torch.from_numpy(x).float().to(device)

        with torch.no_grad():
            y = model(x).detach().cpu().numpy()

        assert y.ndim == 2, \
            "Returned output from the representor must be two dimensional (NC)"
        return y

    return _represent
