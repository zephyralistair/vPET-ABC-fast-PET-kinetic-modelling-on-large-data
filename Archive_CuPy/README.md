# vPET-ABC, Fast PET Kinetic Modelling on Large Data

## Latest Update
Added a module using JAX as backend allowing easier usage.

## Implementation
This code attempts to use a NVIDIA GPU to accelerate the vPET-ABC on FDG compartment models, described in [this paper](https://iopscience.iop.org/article/10.1088/1361-6560/abfa37). The implementation focuses on avoiding for-loops via vectorisations and broadcasting, accelerating computation speed via GPU automated parallelisation, and voxelwisely handling large dataset such as total body PET scans with over 44 million voxels in chunks. It's currently at least 61-fold faster than the [original R code](https://github.com/cgrazian/PETabc) under the same hardware.

GPU implementation relies on the Python module CuPy. Large posterior outputs are currently compressed as HDF5s.
## Usage
### Requirements & Compatibility
This code has been tested on Ubuntu, CentOS 6 (USYD Artemis) and Windows 11. A CUDA-supported NVIDIA GPU is compulsory. A detailed environment currently tested is given in the table below.

| OS                 | Python Version | NVIDIA Graphic Card Driver Version | Cuda Toolkit Version | CuPy Version |
| ------------------ | -------------- | ---------------------------------- | -------------------- | ------------ |
| Ubuntu 20.04.2 LTS | 3.11.4         | 470.223.02                         | 11.4                 | 13.0.0       |
| CentOS 6           | 3.9.15         | Unknown                            | 10.0.130             | 12.2.0       |
| Windows 11         | 3.11.4         | 551.23                             | 11.8                 | 12.2.0       |

### Environment Setup
1. Install a Python environment (>=3.9.15).
2. Install the latest NVIDIA graphic card driver.
3. Install a proper version of the CUDA Toolkit (>=10.0.130, must be compatible with your GPU driver version, see https://docs.nvidia.com/deploy/cuda-compatibility/).
4. Install the corresponding version of CuPy (e.g. CuPy 12.2.0 for CUDA 10.0.130, see https://docs.cupy.dev/en/stable/install.html. Ideally, your CuPy version should be >= 12.2.0, as some methods were not implemented in earlier versions.).
5. Check if you have the latest compatible `pandas`, `numpy`, `scipy`, `tqdm`, `pytables` installed.
6. If you are working on HPCs, certain dependencies might require manual compilation, depending on your environment.

A sample dataset is provided for the 2TCM model, to test the sucessful configuration.
### User Defined Parameters & Models
To use the code, you may modify the parameters and run `python name_of_the_code.py`.
#### Parameters
All user defined parameters are in the `main()` function. A few examples are listed below:

| Name                  | Function                                                                                                                                                              |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data path`           | Path to the input/output data.                                                                                                                                        |
| `seed`                | Random seed for reproducibility.                                                                                                                                      |
| `chunk size`          | Number of voxels to be processed in each batch, maximum size depends on your GPU's memory.                                                                            |
| `S`                   | Number of prior simulations.                                                                                                                                          |
| `thresh`              | Threshold for acceptance, $\text{S}\times\text{thresh}$ is the accepted size                                                                                          |
| `model_0_prob_thres`  | Threshold for model 0 probability, if probability >= threshold, it's model 0.                                                                                         |
| `num_voxel`           | Number of voxels to process. If None, all voxels are.                                                                                                                 |
| `write_paras`         | Flag indicating whether to save parameter posterior.                                                                                                                  |
| `input_compressed`    | Flag indicating whether the input data is compressed (hdf5/csv). Note that if the input data is an HDF5 file, the key used should be "df".                            |
| `output_compressed`   | Flag indicating whether to compress the output posteriors (hdf5/csv). Note that the model probability posterior is always stored as a csv initially, as it is relatively small. Compression will be performed when the computation is finished, if needed. |
| `prior distributions` | Priors distributions of all parameters.                                                                                                                               |

The input data must follow certain structure, an example for the FDG compartment model is shown below:

| frame_length | Ti         | Cb         | 0           | 1           | 2           |
| ------------ | ---------- | ---------- | ----------- | ----------- | ----------- |
| 0            | 0.133333333| 0.632698   | 0.040760215 | 0.036601037 | 0.028408282 |
| 0.166667     | 0.35       | 140.588500 | 18.177458   | 17.420736   | 16.563547   |

where frame_length is the length of the scan frame, Ti is the scan time from injection, Cb is the whole blood input function. Numbers starting from 0 are voxels, storing the corresponding time activity curves.
#### Models
If you wish to create your own models, take a look at function `generate_FDG_models()`. Certain modifications in other functions maybe needed as well, if they involve parameters or i/o. Refer to comments in the code for more details.
## Future Directions
1. Future improvements could include optimising data transfers between the GPU and the CPU, especially if CuPy implements the `scipy.signal.convolve()` function. Meanwhile, we might consider migrating to JAX for better performance.
2. A better data storage method might be to use a database, which is more time-efficient than HDF5, but not necessarily in space. Databases may potentially solve the i/o bottleneck currently deteriorating the program speed significantly when the dataset is gigantic.
3. Implementation of other kinetic models is on the way.
### Common Issues
We will add more common issues and FAQs in the future.
