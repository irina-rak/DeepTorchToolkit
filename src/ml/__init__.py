"""Summary of the ml module for DeepTorchToolkit.
    
    This module provides machine learning utilities and tools for the DeepTorchToolkit library. It includes various functions and classes to facilitate the development, training, and evaluation of deep learning models.

    Modules:
        - datamodules: Functions and classes for data loader and dataset management.
        - models: Predefined neural network architectures and model utilities.

    The file registry.py contains three dictionnaries that holds Datasets, LightningModules
    and LightninDataModules. This is used to easily switch between different models and datasets.

    To add a new model or dataset, simply add a new entry in the corresponding dictionnary and
    create the corresponding class in the models or data module. The new model or dataset will
    be automatically available in the pipeline. Don't forget to import the new class in the
    corresponding LightninModule (lit_<name>.py) for the model.
    To switch between models or datasets, simply change the corresponding entry in the configuration
    file.
"""
