import wandb
from datetime import datetime

class WandbLogger:
    def __init__(self, project_name, api_key):
        wandb.login(key=api_key)
        self.run = wandb.init(project=project_name, name=str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    def log_metrics(self, metrics, step=None):
        """ Log a dictionary of metrics to Weights & Biases. """
        if step is not None:
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)

    def log_image(self, image, caption, step=None):
        """ Log an image with a caption to Weights & Biases. """
        wandb_image = wandb.Image(image, caption=caption)
        if step is not None:
            self.run.log({"retrieval_example": wandb_image}, step=step)
        else:
            self.run.log({"retrieval_example": wandb_image})

    def finish(self):
        """ Finish the W&B run. """
        self.run.finish()