import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_imagenet_probability_augment_configspace():
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


def get_cifar10_probability_augment_configspace():
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
        "brightness_strength", lower=0, upper=1.2, log=False, default_value=0.6,
    )
    contrast_strength = CSH.UniformFloatHyperparameter(
        "contrast_strength", lower=0, upper=1.2, log=False, default_value=0.6,
    )
    saturation_strength = CSH.UniformFloatHyperparameter(
        "saturation_strength", lower=0, upper=1.2, log=False, default_value=0.6,
    )
    hue_strength = CSH.UniformFloatHyperparameter(
        "hue_strength", lower=0, upper=0.4, log=False, default_value=0.2,
    )
    cs.add_hyperparameters([brightness_strength, contrast_strength, saturation_strength, hue_strength])
    return cs
