# DeepTorchToolkit: a modular framework for Deep Learning pipelines

**DeepTorchToolkit** is a lightweight and modular repository for building deep learning pipelines with ease. It provides customizable scripts for supervised learning tasks (e.g., classification, segmentation, object detection) on top of PyTorch Lightning.  

## Features  
- **Customizable Pipelines**: Easily define datasets, models, and training configurations.
- **Task Support**: Pre-built scripts for classification (and soon segmentation!).
- **PyTorch Lightning Powered**: Simplifies training loops and supports distributed training.
- **Logging and Monitoring**: Integrated with Weights & Biases.
- **Extensible Design**: Add custom datasets, models, or loss functions with ease.

---

## Installation  

1. Clone the repository:  
```bash
   git clone https://github.com/yourusername/DeepTorchToolkit.git  
   cd DeepTorchToolkit
```

2. Create a virtual environment (I personally use conda-forge) and install dependencies with Poetry:
```bash
    conda create -n deeptorch python=3.11
    conda activate deeptorch
    poetry install
```


## Usage

1. Define your dataset:
    - Create a new datamodule and dataset classes in `src/ml/datamodules/<dataset_name>/` as shown in the provided templates.
    - `<given_name>_datamodule.py` should contain the dataloaders and pydantic model for configuration.
    - `<given_name>_dataset.py` should contain the dataset class along with the necessary transformations and augmentations.

2. Define your model:
    - Create a new model class and the associated lightning module in `src/ml/models/<model_name>/` as shown in the provided templates.
    - `<given_name>_model.py` should define the model architecture.
    - `lit_<given_name>.py` should contain the lightning module with the forward pass, the training and validation steps, as well as the loss function and metrics.

**Note**: You can use the provided templates as a starting point for your custom datasets and models. Additionally, you can define your custom loss functions, metrics, or callbacks. All custom classes should be placed in the appropriate directories within `src/ml/` that will be automatically imported by the training script from the `registry.py` file located in `src/ml/`. Also, make sure to define the pydantic models for your custom classes, otherwise, the configuration will not be validated.

3. Define your training configuration:
    - Create a new configuration file in `configs/` using the provided templates.  
    - Modify the configuration file to suit your needs.

4. Train your model:
    - Run the training script with the desired configuration file:  
      ```bash
      deeptoolkit train launch configs/classification_template.yaml
      ```


## Contributing

Contributions are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.