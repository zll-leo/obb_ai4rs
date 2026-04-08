"""SwanLab Hook for mmrotate.

This hook logs training and validation metrics to SwanLab.
"""

from typing import Dict, Optional

from mmengine.hooks import LoggerHook
from mmengine.runner import Runner

from mmrotate.registry import HOOKS


@HOOKS.register_module()
class SwanLabHook(LoggerHook):
    """SwanLab hook for logging metrics.

    Args:
        init_kwargs (dict, optional): Arguments passed to swanlab.init.
            Defaults to None.
        interval (int): Logging interval (every n iterations).
            Defaults to 10.
    """

    def __init__(self,
                 init_kwargs: Optional[Dict] = None,
                 interval: int = 10,
                 **kwargs) -> None:
        super().__init__(interval=interval, **kwargs)
        self.init_kwargs = init_kwargs or {}
        self._swanlab = None

    def before_run(self, runner: Runner) -> None:
        """Initialize SwanLab before training.

        Args:
            runner (Runner): The runner of the training process.
        """
        try:
            import swanlab
            self._swanlab = swanlab

            # Set default init kwargs if not provided
            default_init = {
                'project': 'obb-ai4rs',
                'experiment_name': runner.work_dir.split('/')[-1]
                if runner.work_dir else 'train',
            }
            default_init.update(self.init_kwargs)

            swanlab.init(**default_init)
            runner.logger.info('SwanLab initialized successfully.')
        except ImportError:
            runner.logger.warning('SwanLab is not installed. '
                                  'Run: pip install swanlab')

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Dict = None,
                         outputs: Dict = None) -> None:
        """Log metrics after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch.
            data_batch (Dict, optional): Data from dataloader.
            outputs (Dict, optional): Outputs from model.
        """
        if self._swanlab is None:
            return

        if not self.every_n_inner_iters(batch_idx, self.interval):
            return

        # Get learning rate from optimizer
        lr = runner.optim_wrapper.optimizer.param_groups[0]['lr']

        # Prepare log dict
        log_dict = {
            'train/lr': float(lr),
            'train/iter': runner.iter,
        }

        # Get log variables from message_hub runtime_info
        if hasattr(runner, 'message_hub'):
            runtime_info = runner.message_hub.runtime_info
            for key, value in runtime_info.items():
                if isinstance(value, (int, float)):
                    log_dict[f'train/{key}'] = value

        self._swanlab.log(log_dict, step=runner.iter)

    def after_val_epoch(self,
                       runner: Runner,
                       metrics: Dict = None) -> None:
        """Log validation metrics.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict, optional): Metrics from validation.
        """
        if self._swanlab is None or metrics is None:
            return

        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_dict[f'val/{key}'] = value

        if log_dict:
            self._swanlab.log(log_dict, step=runner.iter)

    def after_run(self, runner: Runner) -> None:
        """Finish SwanLab logging.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self._swanlab is not None:
            self._swanlab.finish()
