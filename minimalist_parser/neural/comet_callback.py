from typing import Optional

from allennlp.training import TrainerCallback
from tensorboardX import SummaryWriter

from allennlp.training.callbacks.tensorboard import TensorBoardCallback


@TrainerCallback.register("comet")
class CometCallback(TensorBoardCallback):
    """
    A callback that writes training statistics/metrics to Comet.ml via TensorBoard.
    """

    def __init__(
        self,
        summary_interval: int = 100,
        distribution_interval: Optional[int] = None,
        batch_size_interval: Optional[int] = None,
        should_log_parameter_statistics: bool = False,
        should_log_learning_rate: bool = False,
        use_comet: bool = False,
        comet_api_key: str = None,
        comet_workspace_name: str = None,
        comet_project_name: str = None,
    ) -> None:
        super().__init__(
            "",
            summary_interval=summary_interval,
            distribution_interval=distribution_interval,
            batch_size_interval=batch_size_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
        )

        comet_disabled = not use_comet
        self.comet_config = {
            "api_key": comet_api_key,
            "project_name": comet_project_name,
            "disabled": comet_disabled
        }

        if comet_workspace_name is not None:
            self.comet_config["workspace"] = comet_workspace_name

        self._validation_log = SummaryWriter(comet_config=self.comet_config)
        self._train_log = self._validation_log
