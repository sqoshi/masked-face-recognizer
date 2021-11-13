from .analysis_config import AnalysisConfig, DatasetModifications
from .experiment import Experiment
from .mask_strategy import MaskingStrategy


class AnalysisConfigFactory:
    def __init__(self, path):
        self.dataset_path = path

    def masked_train_set_masked_test_set(self, p: float):
        return AnalysisConfig(
            name=f"{int(p * 100)}p_masked_train_set",
            dataset_path=self.dataset_path,
            split_ratio=0.6,
            train_set_modifications=DatasetModifications(
                mask_ratio=p, inplace=True, mask=MaskingStrategy.alternately
            ),
            test_set_modifications=DatasetModifications(
                mask_ratio=1.0, inplace=True, mask=MaskingStrategy.alternately
            )
        )

    def masked_extended_train_set_masked_test_set(self, p: float):
        return AnalysisConfig(
            name=f"{int(p * 100)}p_masked_extended_train_set",
            dataset_path=self.dataset_path,
            split_ratio=0.6,
            train_set_modifications=DatasetModifications(
                mask_ratio=p, inplace=False, mask=MaskingStrategy.blue
            ),
            test_set_modifications=DatasetModifications(
                mask_ratio=1.0, inplace=True, mask=MaskingStrategy.blue
            )
        )

    def default(self):
        """Config without modifications."""
        return AnalysisConfig(
            name="no_modifications",
            dataset_path=self.dataset_path,
            split_ratio=0.6,
        )

    def masked_test_set(self):
        """Masked test set"""
        return AnalysisConfig(
            name="masked_test_set",
            dataset_path=self.dataset_path,
            split_ratio=0.6,
            test_set_modifications=DatasetModifications(
                mask_ratio=1.0, inplace=True, mask=MaskingStrategy.blue
            )
        )

    def experiment_masking_ratio_influence_inplace(self):
        return Experiment(
            name="masking_ratio_influence_inplace",
            configs=[
                self.masked_train_set_masked_test_set(0.2),
                self.masked_train_set_masked_test_set(0.5),
                self.masked_train_set_masked_test_set(0.7),
            ]
        )

    def experiment_masking_ratio_influence_extended(self):
        return Experiment(
            name="masking_ratio_influence_extended",
            configs=[
                self.masked_train_set_masked_test_set(0.2),
                self.masked_train_set_masked_test_set(0.5),
                self.masked_train_set_masked_test_set(0.7),
            ]
        )

    def research_path(self):
        return [
            self.default(),
            self.masked_test_set(),
            self.experiment_masking_ratio_influence_inplace(),
            self.experiment_masking_ratio_influence_extended()
            # self.masked_train_set_masked_test_set(0.2),  # g1
            # self.masked_train_set_masked_test_set(0.5),  # g1
            # self.masked_train_set_masked_test_set(0.7),  # g1
            # self.masked_extended_train_set_masked_test_set(0.2),
            # self.masked_extended_train_set_masked_test_set(0.5),
            # self.masked_extended_train_set_masked_test_set(0.7),
        ]
