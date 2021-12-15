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
            personal_images_quantity=30,
            identities_limit=50,
            equal_piqs=True,
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
            personal_images_quantity=30,
            identities_limit=50,
            equal_piqs=True,
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
            # personal_images_quantity=20,
            # identities_limit=50,
            # equal_piqs=True
        )

    def my_test_svm_config(self, svm_config):
        """Config without modifications."""
        return AnalysisConfig(
            name="my_test_svm_config",
            dataset_path=self.dataset_path,
            split_ratio=0.7,
            personal_images_quantity=30,
            identities_limit=50,
            equal_piqs=True,
            landmarks_detection=False,
            svm_config=svm_config,
            train_set_modifications=DatasetModifications(
                mask_ratio=1.0, inplace=False, mask=MaskingStrategy.blue
            ),
            test_set_modifications=DatasetModifications(
                mask_ratio=1.0, inplace=True, mask=MaskingStrategy.blue
            )
        )

    def my_test(self, m1, m2, svm_config=None):
        """Config without modifications."""
        return AnalysisConfig(
            name="no_modifications",
            dataset_path=self.dataset_path,
            split_ratio=0.7,
            personal_images_quantity=30,
            identities_limit=50,
            equal_piqs=True,
            landmarks_detection=False,
            train_set_modifications=DatasetModifications(
                mask_ratio=1.0, inplace=False, mask=m1
            ),
            test_set_modifications=DatasetModifications(
                mask_ratio=1.0, inplace=True, mask=m2
            )
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

    def kernel_experiment(self):
        def create_svm_config(kernel="linear", degree=1, c=1.0):
            return {"C": c, "kernel": kernel,
                    "degree": degree,
                    "probability": True,
                    "random_state": True}

        return Experiment(
            name="experiment_kernels_influence2",
            configs=
            [self.my_test_svm_config(create_svm_config(kernel="sigmoid"))]
            # [self.my_test_svm_config(create_svm_config())] +
            # [self.my_test_svm_config(create_svm_config(kernel="rbf"))] +
            # [self.my_test_svm_config(create_svm_config(kernel="poly", degree=i)) for i in
            #  [3, 4, 5, 7, 10]]
        )

    def experiment_masking_ratio_influence_inplace(self):
        return Experiment(
            name="experiment_masking_ratio_influence_inplace",
            configs=[
                self.masked_train_set_masked_test_set(0.2),
                self.masked_train_set_masked_test_set(0.5),
                self.masked_train_set_masked_test_set(0.7),
                self.masked_train_set_masked_test_set(1.0),
            ]
        )

    def experiment_masking_ratio_influence_extended(self):
        return Experiment(
            name="experiment_masking_ratio_influence_extended",
            configs=[
                self.masked_train_set_masked_test_set(0.2),
                self.masked_train_set_masked_test_set(0.5),
                self.masked_train_set_masked_test_set(0.7),
                self.masked_train_set_masked_test_set(1.0),
            ]
        )

    def experiment_mask_types(self):
        return Experiment(
            name="experiment_masking_ratio_influence_extended",
            configs=[
                self.my_test(MaskingStrategy.blue, MaskingStrategy.grey),
                self.my_test(MaskingStrategy.alternately, MaskingStrategy.alternately),
                self.my_test(MaskingStrategy.black_box, MaskingStrategy.alternately),
            ]
        )

    def analyze_unknown_personality_influence(self):
        default_config = self.masked_train_set_masked_test_set(0.7)
        default_config.name = "skipped_personality"
        default_config.skip_unknown = True
        return default_config

    def analyze_alternate_mask(self):
        default_config = self.masked_test_set()
        default_config.name = "different_masks"
        default_config.modifications.train = DatasetModifications(
            mask_ratio=0.7, inplace=False, mask=MaskingStrategy.alternately
        )
        return default_config

    def analyze_black_boxes_instead_of_mask_influence(self):
        default_config = self.masked_test_set()
        default_config.name = "black_boxes"
        default_config.modifications.train = DatasetModifications(
            mask_ratio=0.7, inplace=False, mask=MaskingStrategy.black_box
        )
        return default_config

    def research_path(self):
        return [
            self.kernel_experiment(),
            # self.experiment_masking_ratio_influence_inplace(),
            # self.experiment_mask_types(),
            # self.default(),
            #    self.masked_test_set(),
            #    self.experiment_masking_ratio_influence_inplace(),
            #    self.experiment_masking_ratio_influence_extended(),
            #    self.analyze_unknown_personality_influence(),
            #    self.analyze_alternate_mask(),
            #    self.analyze_black_boxes_instead_of_mask_influence()
        ]
