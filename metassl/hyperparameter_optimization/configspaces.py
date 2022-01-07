import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_imagenet_probability_simsiam_augment_configspace():
    cs = CS.ConfigurationSpace()
    p_colorjitter = CSH.UniformFloatHyperparameter(
        "p_colorjitter", lower=0, upper=1, log=False, default_value=0.8,
    )
    p_grayscale = CSH.UniformFloatHyperparameter(
        "p_grayscale", lower=0, upper=1, log=False, default_value=0.2,
    )
    p_gaussianblur = CSH.UniformFloatHyperparameter(
        "p_gaussianblur", lower=0, upper=1, log=False, default_value=0.5,
    )
    cs.add_hyperparameters([p_colorjitter, p_grayscale, p_gaussianblur])
    return cs


def get_cifar10_probability_simsiam_augment_configspace():
    cs = CS.ConfigurationSpace()
    p_colorjitter = CSH.UniformFloatHyperparameter(
        "p_colorjitter", lower=0, upper=1, log=False, default_value=0.8,
    )
    p_grayscale = CSH.UniformFloatHyperparameter(
        "p_grayscale", lower=0, upper=1, log=False, default_value=0.2,
    )
    cs.add_hyperparameters([p_colorjitter, p_grayscale])
    return cs

def get_color_jitter_strengths_configspace():
    cs = CS.ConfigurationSpace()
    brightness_strength = CSH.UniformFloatHyperparameter(
        "brightness_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    contrast_strength = CSH.UniformFloatHyperparameter(
        "contrast_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    saturation_strength = CSH.UniformFloatHyperparameter(
        "saturation_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    hue_strength = CSH.UniformFloatHyperparameter(
        "hue_strength", lower=0, upper=0.4, log=False, default_value=0.1,
    )
    cs.add_hyperparameters([brightness_strength, contrast_strength, saturation_strength, hue_strength])
    return cs


def get_rand_augment_configspace():
    cs = CS.ConfigurationSpace()
    num_ops = CSH.UniformIntegerHyperparameter(
        "num_ops", lower=1, upper=3, log=False, default_value=2,
    )
    magnitude = CSH.UniformIntegerHyperparameter(
        "magnitude", lower=0, upper=30, log=False, default_value=15,
    )
    cs.add_hyperparameters([num_ops, magnitude])
    return cs


def get_probability_augment_configspace():
    cs = CS.ConfigurationSpace()
    p_color_transformations = CSH.UniformFloatHyperparameter(
        "p_color_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_geometric_transformations = CSH.UniformFloatHyperparameter(
        "p_geometric_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_non_rigid_transformations = CSH.UniformFloatHyperparameter(
        "p_non_rigid_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_quality_transformations = CSH.UniformFloatHyperparameter(
        "p_quality_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_exotic_transformations = CSH.UniformFloatHyperparameter(
        "p_exotic_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    cs.add_hyperparameters(
        [p_color_transformations, p_geometric_transformations, p_non_rigid_transformations, p_quality_transformations, p_exotic_transformations]

    )
    return cs