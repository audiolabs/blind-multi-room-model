from typing import Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from webdataset import WebDataset


class BlindSlopesDataModule(LightningDataModule):
    """Baseline datamodule"""

    def __init__(
        self,
        train_url: str,
        valid_url: str,
        test_url: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers

        out_tuple = (
            "signal.pyd",
            "edc.pyd",
            # "norm.pyd",
            "pos.pyd",
            "ls_id.pyd",
            "scene_id.pyd",
            "t60.pyd",
            "edc_dfn.pyd",
        )

        # dataset instances, load input signal and all labels
        train_dataset = WebDataset(train_url).decode("pil").to_tuple(*out_tuple)
        valid_dataset = WebDataset(valid_url).decode("pil").to_tuple(*out_tuple)
        test_dataset = WebDataset(test_url).decode("pil").to_tuple(*out_tuple)

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
        )

        self.valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.valid_loader

    def test_dataloader(self) -> DataLoader:
        return self.test_loader

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        pass
