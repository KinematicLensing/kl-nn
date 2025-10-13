# kl-nn
Neural network tools for accelerating KL analyses

### Setup
1. Git clone [kl-tools](https://github.com/wxs0703/kl-tools), move to the `data_generate` branch
2. Create a new conda environment, and install all packages specified in the `environments.yml` file in the kl-tools directory
3. Do `pip install .` inside the kl-tools directory to install kl-tools in your environment
4. Install [ml-pyxis](https://github.com/vicolab/ml-pyxis) for database generation

### Note: Before running any slurm scripts, make sure to change any directories involved in the script itself as well as any scrips that it calls!

### Data Generation
Code for generating training data is located in `kl-nn/data_generate`

#### The steps to generating data are as follows:
1. Generate data vector samples using `latin_hypercube.py`. This will generate a csv file with all the data vectors to generate training data from. Number of samples and parameter ranges can be configured in file. Training and testing samples are generated with two separate runs.
2. Generate fits files for each data vector.
    a. make sure the file directories in the `generate_fits.py`, `generate_training_wrapper.py`, and `generate_testing_wrapper.py` files are correct. This process will be simplified in future updates.
    b. training and testing fits files are generated using the `generate_train_set.slurm` and `generate_test_set.slurm` scripts respectively. Compute resources and how each parallel job is split up can be configured in the script.
    c. `check_completeness.ipynb` and `generate_leftovers.py` are diagnostic scripts in case step b does not generate the entire sample size. This could happen if requested job time is not enough to generate everything.
3. Create training and testing databases using `make_database.ipynb`. Use the `_only_g` version of the notebook if you only want to train to predict shear. The database format is smaller and easier for the training algorithm to digest.

### Network Config, Training and Testing
Code for configuring neural network, training and testing is located in `kl-nn/arch`

Network configuration is all done in `networks.py`. Loss function and training process can be edited in `train.py`.

Training configuration is done in `config.py`. Important parameters are `'size', 'pars_dir', 'data_dir'` as well as all the parameters in the `train` dictionary. To train simply configure and run `train_model_full.slurm`. the notebook `train_model.ipynb` only exists for debug purposes.

To test the network simply follow the `test_model.ipynb` notebook.