import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def get_optimizer_and_scheduler(model, optimizer_params, scheduler_params):
    """
    Returns the optimizer and scheduler based on the configuration.

    Args:
        model (torch.nn.Module): The model to optimize.
        optimizer_params (dict): Parameters for the optimizer.
            Example:
            {
                "type": "SGD",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 5e-4
            }
        scheduler_params (dict): Parameters for the scheduler.
            Example for StepLR:
            {
                "type": "StepLR",
                "step_size": 10,
                "gamma": 0.1
            }
            Example for ReduceLROnPlateau:
            {
                "type": "ReduceLROnPlateau",
                "mode": "min",
                "factor": 0.1,
                "patience": 5
            }

    Returns:
        tuple: (optimizer, scheduler)
    """
    # Select optimizer
    if optimizer_params["type"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_params["lr"],
            momentum=optimizer_params.get("momentum", 0),
            weight_decay=optimizer_params.get("weight_decay", 0)
        )
    elif optimizer_params["type"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_params["lr"],
            weight_decay=optimizer_params.get("weight_decay", 0)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_params['type']}")

    # Select scheduler
    # Select scheduler if provided
    scheduler = None
    if scheduler_params:
        if scheduler_params["type"] == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_params["step_size"],
                gamma=scheduler_params["gamma"]
            )
        elif scheduler_params["type"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_params["mode"],
                factor=scheduler_params["factor"],
                patience=scheduler_params["patience"]
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_params['type']}")

    return optimizer, scheduler
