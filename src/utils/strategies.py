"""DDP process group without the DistributedDataParallel wrapper: pilot 1 averages gradients by hand (Amendment 1 S16) and keeps the bank outside the module.

Usage: `trainer=Trainer(strategy=ManualSyncDDPStrategy(), ...)` (configs/trainer/p1_ddp.yaml).
"""
from lightning.pytorch.overrides.distributed import _sync_module_states
from lightning.pytorch.strategies import DDPStrategy


class ManualSyncDDPStrategy(DDPStrategy):
    def configure_ddp(self) -> None:
        # One broadcast of rank 0's weights replaces the wrapper's start-up sync.
        _sync_module_states(self.model)
