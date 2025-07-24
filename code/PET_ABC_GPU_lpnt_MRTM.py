"""
==================================
// LICENSE:
// Copyright 2024 University of Sydney
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
// ===============================
// AUTHOR       : Qinlin (Alistair) Gu, 
//                modified based on Dr. Clara Grazian's original R code.
// CREATE DATE  : 26/03/2024
// PURPOSE      : To perform the vABC algorithm using GPU acceleration.
// SPECIAL NOTES:
// ===============================
// Change History:
// 20/06/2024 - Qinlin (Alistair) Gu - Copied from the 2TCM version and modified
//                                     for the lpnt/MRTM version.
==================================
"""

import cupy as cp
import pandas as pd
import numpy as np
import os
import time
import warnings
from tqdm import tqdm

# Filter out FutureWarnings in Pandas
warnings.filterwarnings("ignore", category=FutureWarning)

def extract_values(df):
    """
    Extracts the values from the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple: A tuple containing the extracted values.
            Each has a shape of (num_time_frame).
    """
    time_frame_size = cp.array(df.iloc[:, 0]).astype(cp.float32)
    Ti = cp.array(df.iloc[:, 1].values).astype(cp.float32)
    Cr = cp.array(df.iloc[:, 2].values).astype(cp.float32)

    return time_frame_size, Cr, Ti

def extract_TAC_chunks(df, index, chunk_size, num_voxel):
    """
    Extracts chunks of TAC (Time-Activity Curve) from the given DataFrame.
    Chunks are used so that only a proportion of the data is loaded into memory 
        at a time, preventing memory overflow.

    Args:
        df (pd.DataFrame): Input DataFrame.
        index (int): Starting index of the chunk.
        chunk_size (int): Size of each chunk.
        num_voxel (int): Number of voxels.

    Returns:
        cp.ndarray: Extracted TAC chunks, shaped as (num_time_frame, num_voxel).
    """
    df_column_size = df.shape[1]
    num_of_other_columns = 3 ## 3 columns are for time_frame_size, Cb, and Ti
    if num_voxel is None: ## When None, use all voxels
        num_voxel = df_column_size - num_of_other_columns
    chunk_end = min(index + chunk_size, df_column_size, num_voxel + num_of_other_columns)
    Ct = cp.array(df.iloc[:, index: chunk_end].values).astype(cp.float32)

    return Ct

def output_file_init(path_output_para, path_output_model, write_paras, output_compressed):
    """
    Initializes the output files for parameter and model data, so that results
        can be recorded in chunks.

    Args:
        path_output_para (str): Path to the parameter output file.
        path_output_model (str): Path to the model output file.
        write_paras (bool): flag indicating whether to write parameter posterior.
        output_compressed (bool): flag indicating whether to compress the output 
                                  posteriors (hdf5/csv). Note that the model 
                                  probability posterior is always stored as a csv
                                  as it is relatively small.
    """
    para_columns = ["TAC_No", "R_1", "K_2", "K_2a", 
                    "gamma", "t_D", "t_P", "alpha", "model"]
    num_of_columns = len(para_columns)
    model_p_columns = ["Voxel_No", "model", "probability_of_model"]

    if write_paras:
        para_df = pd.DataFrame(columns = para_columns)
        para_df.iloc[:, 0] = para_df.iloc[:, 0].astype(np.int32)
        para_df.iloc[:, 1:num_of_columns-1] = para_df.iloc[:, 1:num_of_columns-1].astype(np.float32)
        para_df.iloc[:, -1] = para_df.iloc[:, -1].astype(np.str_)
        if output_compressed:
            para_df.to_hdf(path_output_para, 
                           key = "column_names", 
                           format='table', 
                           index = False, 
                           mode = "w", 
                           complevel = 9)
        else:
            para_df.to_csv(path_output_para, 
                           index = False, 
                           mode = "w", 
                           # compression = "xz"
                           )

    model_p_df = pd.DataFrame(columns = model_p_columns)
    model_p_df.to_csv( ## csv first, compression later, otherwise slow
        path_output_model.replace("h5", "csv"), 
        index = False, 
        mode = "w", 
        # compression = "xz"
        )

def output_dataframe(para, model_p, write_paras):
    """
    Prepares the output DataFrames for parameter and model data.

    Args:
        para (cp.ndarray): Parameter data. Contains Vb, alpha1, alpha2, theta1,
            theta2, and model values.
        model_p (cp.ndarray): Model data.
        write_paras (bool): flag indicating whether to write parameter posterior.

    Returns:
        tuple: A tuple containing the prepared DataFrames.
            para_df is the parameter posterior DataFrame. (K_1 etc.)
            model_p_df is the model probability posterior DataFrame.
    """
    if write_paras:
        para_df = pd.DataFrame(para)
        para_df[0] = para_df.iloc[:, 0].astype(int)

        cols_to_replace_as_NA_indices = [4, 5, 6, 7]
        para_df.loc[
            para_df.iloc[:, -1] == 'MRTM', 
            para_df.columns[cols_to_replace_as_NA_indices]
            ] = pd.NA

        para_df.iloc[:, -1] = para_df.iloc[:, -1].replace({0: 'MRTM', 
                                                           1: 'lp-nt'})
    else:
        para_df = None

    model_p_df = pd.DataFrame(model_p)
    model_p_df[0] = model_p_df.iloc[:, 0].astype(int)
    model_p_df[2] = model_p_df.iloc[:, 2].astype(float)
    model_p_df.iloc[:, 2] = np.where(model_p_df.iloc[:, 1] == 1, 
                                     1 - model_p_df.iloc[:, 2], 
                                     model_p_df.iloc[:, 2])
    model_p_df.iloc[:, 1] = model_p_df.iloc[:, 1].replace({0: 'MRTM', 
                                                           1: 'lp-nt'})

    return para_df, model_p_df

def write_csv_chunks(para_df, 
                     model_p_df, 
                     path_output_para, 
                     path_output_model, 
                     write_paras, 
                     output_compressed):
    """
    Writes the parameter and model data to CSV files in chunks.

    Args:
        para_df (pd.DataFrame): Parameter data.
        model_p_df (pd.DataFrame): Model data.
        path_output_para (str): Path to the parameter output file.
        path_output_model (str): Path to the model output file.
        write_paras (bool): flag indicating whether to write parameter posterior.
        output_compressed (bool): flag indicating whether to compress the output 
                                  posteriors (hdf5/csv). Note that the model 
                                  probability posterior is always stored as a csv
                                  as it is relatively small.
    """
    if write_paras:
        para_columns = ["TAC_No", "R_1", "K_2", "K_2a", 
                        "gamma", "t_D", "t_P", "alpha", "model"]
        num_of_columns = len(para_columns)
        para_df.columns = para_columns

        unique_voxels = para_df.iloc[:, 0].unique()
        min_index = np.min(unique_voxels)
        max_index = np.max(unique_voxels)
        para_df.iloc[:, 0] = para_df.iloc[:, 0].astype(np.int32)
        para_df.iloc[:, 1:num_of_columns-1] = para_df.iloc[:, 1:num_of_columns-1].astype(np.float32)
        para_df.iloc[:, -1] = para_df.iloc[:, -1].astype(np.str_)
        if output_compressed:
            para_df.to_hdf(path_output_para, 
                        key = f"voxel_{min_index}_{max_index}", 
                        format='table', 
                        index = False, 
                        mode = 'a', 
                        complevel = 9, 
                        min_itemsize = {'model': 11})
        else:
            para_df.to_csv(
                path_output_para, 
                header = False, 
                index = False, 
                mode = "a", 
                # compression = "xz"
                )

    model_p_df.to_csv(
        path_output_model.replace("h5", "csv"), 
        header = False, 
        index = False, 
        mode = "a", 
        # compression = "xz"
        )
    
def compress_csv(path_output_model):
    """
    Compresses the model output CSV file to HDF5 format, when all are done

    Args:
        path_output_model (str): Path to the model output file.
    """
    path_output_model_csv = path_output_model.replace("h5", "csv")
    df = pd.read_csv(path_output_model_csv)
    df.iloc[:, 0] = df.iloc[:, 0].astype(np.int32)
    df.iloc[:, 1] = df.iloc[:, 1].astype(np.str_)
    df.iloc[:, 2] = df.iloc[:, 2].astype(np.float32)
    df.to_hdf(path_output_model, 
              key = "df", 
              index = False, 
              mode = "w", 
              complevel = 9)

    if os.path.exists(path_output_model_csv):
        os.remove(path_output_model_csv)

def get_Ct(time_frame_size, Cr, Cr_cumsum, Ct, Ct_cumsum_neg, Ti, paras):
    """
    Generates FDG (Fluorodeoxyglucose) model TACs using simulated priors.

    Args:
        time_frame_size (cp.ndarray): Time frame size.
        Cr (cp.ndarray): Cr values.
        Cr_cumsum (cp.ndarray): Cumulative sum of Cr values.
        Ct (cp.ndarray): Ct values.
        Ct_cumsum_neg (cp.ndarray): Negative cumulative sum of Ct values.
        Ti (cp.ndarray): Ti values.
        paras (cp.ndarray): Parameter values. Contains Vb, alpha1, alpha2,
            theta1, theta2, and model values.

    Returns:
        cp.ndarray: model TACs.
    """
    R1, K2, K2a, gamma, tD, tP, alpha, model = [row for row in paras]

    Ind = (Ti - tD) > 0
    ht = cp.maximum(0, (Ti - tD) / (tP - tD)) ** alpha * \
        cp.exp(alpha * (1 - (Ti - tD) / (tP - tD))) * Ind

    ## model mask
    model_mask = model.squeeze() == 0

    # if model == 1:
    theta = cp.stack([R1, K2, K2a, gamma], axis = 2)
    # if model == 0:
    model_mask = model_mask[None, :, None]
    theta[:, :, -1, :][model_mask] = 0
    mask_shape = cp.broadcast(cp.empty(Ct.shape, dtype = cp.float32), 
                              cp.empty(ht.shape, dtype = cp.float32)).shape
    model_mask = cp.broadcast_to(model_mask, mask_shape)
    Ct_ht = cp.where(model_mask, 0, -cp.cumsum(Ct * ht * time_frame_size, axis = -1))

    target_shape = Ct_ht.shape

    Cr_broadcasted = cp.broadcast_to(Cr, target_shape)
    Cr_cumsum_broadcasted = cp.broadcast_to(Cr_cumsum, target_shape)
    Ct_cumsum_neg_broadcasted = cp.broadcast_to(Ct_cumsum_neg, target_shape)
    # Ct_ht already has the desired shape, so no need to broadcast

    # Now stack them along the fourth dimension
    BigMat = cp.stack((Cr_broadcasted, 
                       Cr_cumsum_broadcasted, 
                       Ct_cumsum_neg_broadcasted, 
                       Ct_ht), 
                       axis = -1)

    bigmat_shape = BigMat.shape
    theta_shape = (bigmat_shape[0], bigmat_shape[1], bigmat_shape[3], 1)
    theta = cp.broadcast_to(theta, theta_shape)

    M = cp.einsum('ijkl,ijln->ijkn', BigMat, theta)

    return M.squeeze(axis = -1)

def generate_models(time_frame_size, Cr, Cr_cumsum, Ct, Ct_cumsum, Ti, par_mat):
    """
    A wrapper function to generate FDG models using the given parameters,
    including fitting the input function for minimal convolution error.

    Args:
        time_frame_size (cp.ndarray): Time frame size.
        Cr (cp.ndarray): Cr values.
        Cr_cumsum (cp.ndarray): Cumulative sum of Cr values.
        Ct (cp.ndarray): Ct values.
        Ct_cumsum (cp.ndarray): Cumulative sum of Ct values.
        Ti (cp.ndarray): Ti values.
        par_mat (cp.ndarray): Parameter matrix.

    Returns:
        cp.ndarray: FDG model TACs using the prior simulations.
    """

    time_frame_size = time_frame_size[None, None, :]
    Cr = Cr[None, None, :]
    Cr_cumsum = Cr_cumsum[None, None, :]
    Ct = Ct.T[:, None, :]
    Ct_cumsum_neg = -Ct_cumsum.T[:, None, :]
    Ti = Ti[None, None, :]
    ## shape (1, 1, num_time_frame)
    ## to match (num_vox, num_prior_simulation_size, num_time_frame)
    paras = par_mat.T[:, None, :, None]
    ## shape (num_variable, 1, num_prior_simulation_size, 1)
    ## to match (num_variable, num_vox, num_prior_simulation_size, num_time_frame)

    M = get_Ct(time_frame_size, Cr, Cr_cumsum, Ct, Ct_cumsum_neg, Ti, paras)

    return M

def distance_function(M, Ct, distance_type, validation_mode=False, 
                      hyperparameter=None):
    """
    Calculates the distance function between the model and the observed data.
    
    Args:
        M (cp.ndarray): FDG model TACs using the prior simulations.
            (num_vox, num_prior_simulation_size, num_time_frame)
        Ct (cp.ndarray): TAC chunks.
            (num_vox, 1, num_time_frame)
        distance_type (str): Type of distance function to use. Options are:
            "L1", "L2", "Cauchy", "Huber", "Welsch", "CvM".
            L1: L1 distance. L1 norm is the sum of the absolute values of the 
                             vector.
            L2: L2 distance. L2 norm is the square root of the sum of the 
                             squared values of the vector.
            Cauchy: Cauchy distance. Cauchy loss is the sum of the logarithm of 
                                     1 plus the square of the vector divided by 
                                     gamma.
            Huber: Huber distance. Huber loss is the convolution of the absolute
                                   value function with the rectangular function,
                                   scaled and translated.
            Welsch: Welsch distance. Welsch loss is the sum of 1 minus the 
                                     exponential of the square of the vector 
                                     divided by gamma.
            CvM: Cramer-von Mises distance. CvM norm is the sum of the square of
                                            the ranks of the vector.
        validation_mode (bool): flag indicating whether to use validation mode.
        hyperparameter (float): Hyperparameter value for the distance function.
            Tunable.

    Returns:
        cp.ndarray: Errors calculated using the given distance function.
    """
    if distance_type == "L1":
        errors = cp.sum(cp.abs(M - Ct), axis = -1)
    elif distance_type == "L2":
        errors = cp.sum(cp.square(M - Ct), axis = -1)
    elif distance_type == "Cauchy":
        gamma = hyperparameter if validation_mode else 19320.175439
        errors = cp.sum(cp.log(1 + cp.square((M - Ct) / gamma)), axis = -1)
    elif distance_type == "Huber":
        delta = hyperparameter if validation_mode else 10925.438596
        errors = cp.sum(cp.where(cp.abs(M - Ct) <= delta, 
                                 0.5 * cp.square(M - Ct), 
                                 delta * (cp.abs(M - Ct) - 0.5 * delta)), 
                        axis = -1)
    elif distance_type == "Welsch":
        gamma = hyperparameter if validation_mode else 19701.754386
        errors = cp.sum(1 - cp.exp(-cp.square((M - Ct) / gamma)), axis = -1)
    elif distance_type == "CvM":
        M_sorted = cp.sort(M, axis = -1)
        Ct_sorted = cp.sort(Ct, axis = -1)
        shape = (Ct_sorted.shape[0], M_sorted.shape[1], M_sorted.shape[2])
        M_sorted = cp.broadcast_to(M_sorted, shape)
        Ct_sorted = cp.broadcast_to(Ct_sorted, shape)
        combined_sorted = cp.sort(
            cp.concatenate((M_sorted, Ct_sorted), axis = -1), 
            axis = -1
            )
        ranks_M = cp.sum(
            combined_sorted[..., None, :] < M_sorted[..., None], 
            axis = -1
        )
        ranks_Ct = cp.sum(
            combined_sorted[..., None, :] < Ct_sorted[..., None],
            axis = -1
        )
        M = M.shape[-1]
        N = Ct.shape[-1]
        i_list = cp.arange(M)
        j_list = cp.arange(N)
        U = M * cp.sum(cp.square(ranks_M - i_list), axis = -1) + \
            N * cp.sum(cp.square(ranks_Ct - j_list), axis = -1)
        errors = U / (M * N * (M + N)) - (4 * M * N - 1) / (6 * (M + N))
        errors = errors + cp.random.normal(0, 0.0001, errors.shape)
    # elif distance_type == "normalised_L1":
    #     errors = cp.sum(cp.abs(M - Ct) / cp.abs(Ct), axis = -1)
    else:
        return distance_function(M, Ct, "L1")

    return errors

def calculate_results(M, par_mat, Ct, S, thresh, write_paras, 
                      model_0_prob_thres=0.5, vox_num_start=0, 
                      distance_type="L1", validation_mode=False, 
                      hyperparameter=None):
    """
    Calculates the accepted simulations based on the given inputs.

    Args:
        M (cp.ndarray): FDG model TACs using the prior simulations.
            (num_vox, num_prior_simulation_size, num_time_frame)
        par_mat (cp.ndarray): Parameter matrix.
            (num_prior_simulation_size, num_variable)
        Ct (cp.ndarray): TAC chunks.
            (num_time_frame, num_vox)
        S (int): Prior simulation size. i.e. num_prior_simulation_size
        thresh (float): Threshold value for acceptance. Tunable.
        write_paras (bool): flag indicating whether to write parameter posterior.
        model_0_prob_thres (float): Threshold for model 0 probability. Tunable.
        vox_num_start (int): Starting voxel number. For batching purpose.
        distance_type (str): Type of distance function to use. Options are:
            "L1", "L2", "Cauchy", "Huber", "Welsch", "CvM".
        validation_mode (bool): flag indicating whether to use validation mode.
        hyperparameter (float): Hyperparameter value for the distance function.
            Tunable.

    Returns:
        tuple: A tuple containing the accepted parameter posteriors and model 
            probabilities.
    """
    num_vox = Ct.shape[-1]
    num_variable = 8

    voxel_numbers = cp.arange(num_vox) + vox_num_start

    Ct = Ct.T[:, None, :] ## (num_vox, 1, num_time_frame), 
                          ## second dimension for broadcasting
    errors = distance_function(
        M, 
        Ct, 
        distance_type = distance_type, 
        validation_mode = validation_mode, 
        hyperparameter = hyperparameter
        )
    ## calculate errors along time_frame axis
    ## (num_vox, num_prior_simulation_size)
    h = cp.quantile(errors, thresh, axis = -1) ## along num_prior_simulation_size axis
                                               ## (num_vox)
    accepted_mask = errors <= h[:, None] ## (num_vox, num_prior_simulation_size)
    accepted_size = int(cp.count_nonzero(accepted_mask[0]))
    ## This usually is fine but is risky
    ## Sometimes for special distance functions, there can be ties
    ## Leading to different accepted_size
    ## Making the vectorisation of the code not work

    ## was (num_prior_simulation_size, num_variable)
    par_mat_broadcast_shape = (num_vox, S, num_variable)
    par_mat = cp.broadcast_to(par_mat, par_mat_broadcast_shape)
    ## to repeat the par_mat for each voxel, for output purpose
    ## (num_vox, num_prior_simulation_size, num_variable)
    accepted_errors = errors[accepted_mask]
    accepted_errors = accepted_errors.reshape(num_vox, accepted_size)

    accepted_mask = cp.broadcast_to(accepted_mask[:, :, None], 
                                    par_mat_broadcast_shape)
    ## mask was (num_vox, num_prior_simulation_size)
    ## mask repeated for each variable
    ## (num_vox, num_prior_simulation_size, num_variable)
    accepted = par_mat[accepted_mask]
    accepted = accepted.reshape(num_vox, accepted_size, num_variable)
    ## reshaping needed because applying the mask will flatten the array
    ## (num_vox, accepted_size, num_variable)
    ## Errors will happen if accepted_size is different for different voxels
    ## Potential risky distance functions include those using ranks

    # models = accepted[:, :, -1] ## (num_vox, accepted_size)
    #                             ## array of models accepted for each voxel
    # percentage_zeros = cp.mean(models == 0, axis = -1) ## along accepted_size axis
    # models = (percentage_zeros < model_0_prob_thres).astype(cp.int32)
    # model_p = cp.column_stack((voxel_numbers, models, percentage_zeros))

    if not write_paras:
        accepted = None

    return accepted, accepted_errors

def vABC(num_voxel, path_data, path_output_para, path_output_model, par_mat, S, 
         thresh, model_0_prob_thres, write_paras, input_compressed=False, 
         output_compressed=False, chunk_size=25, 
         distance_type="L1", validation_mode=False, hyperparameter=None):
    """
    Performs the vABC (Variational Approximate Bayesian Computation) algorithm.

    Args:
        num_voxel (int): Number of voxels to process. If None, all voxels are
        path_data (str): Path to the input data file.
        path_output_para (str): Path to the parameter posterior output file.
        path_output_model (str): Path to the model probability posterior output file.
        par_mat (cp.ndarray): Parameter matrix, size (num_prior_simulation_size, num_variable).
        S (int): Size of the simulation. i.e. num_prior_simulation_size
        thresh (float): Threshold value for acceptance. Tunable.
        model_0_prob_thres (float): Threshold for model 0 probability. Tunable.
        write_paras (bool): flag indicating whether to write parameter posterior.
        input_compressed (bool): flag indicating whether the input data is compressed (hdf5/csv).
        output_compressed (bool): flag indicating whether to compress the output 
                                  posteriors (hdf5/csv). Note that the model 
                                  probability posterior is always stored as a csv
                                  as it is relatively small.
        chunk_size (int): Size of each chunk. Used to prevent memory overflow.
        distance_type (str): Type of distance function to use. Options are:
            "L1", "L2", "Cauchy", "Huber", "Welsch", "CvM".
        validation_mode (bool): flag indicating whether to use validation mode.
        hyperparameter (float): Hyperparameter value for the distance function.
            Tunable.
    """
    if input_compressed:
        df = pd.read_hdf(path_data, "df") ## alter, use read_csv(chunksize=)
    else:
        df = pd.read_csv(path_data)
    time_frame_size, Cr, Ti = extract_values(df)
    Cr_cumsum = cp.cumsum(Cr * time_frame_size, axis = 0)
    print("Data extracted...")

    index = 3 ## ignoring the first 3 columns
              ## which are for time_frame_size, Cb, and Ti
    df_column_size = df.shape[1] ## number of columns in the DataFrame

    if thresh < 0.01:
        par_chunk_num = 1000
    else:
        par_chunk_num = 1
    chunked_par_mat = cp.array_split(par_mat, par_chunk_num, axis = 0)

    output_file_init(
        path_output_para, 
        path_output_model, 
        write_paras, 
        output_compressed
        )
    ## initialise the output files
    print("Output files initialised...")

    if num_voxel is None: ## When None, use all voxels
        num_voxel = df_column_size - 3

    # Calculate the number of iterations needed for the tqdm progress bar
    total_iterations = min(num_voxel, df_column_size - 3) / chunk_size
    total_iterations = int(total_iterations) if total_iterations.is_integer() else int(total_iterations) + 1

    voxel_timing = []  # To store the time for each voxel or batch
    gpu_start_event = cp.cuda.Event()
    gpu_end_event = cp.cuda.Event()

    for _ in tqdm(range(total_iterations)):
        ## batching to prevent memory overflow
        if index >= df_column_size or index >= num_voxel + 3:
            break

        # Start the CPU timer (wall clock time)
        cpu_start_wall = time.time()
        # Start the CPU timer (total CPU time for all cores)
        cpu_start_process = os.times()
        # Start the GPU timer
        gpu_start_event.record()

        Ct = extract_TAC_chunks(df, index, chunk_size, num_voxel)
        num_vox = Ct.shape[-1]
        vox_num_start = index - 3
        voxel_numbers = cp.arange(num_vox) + vox_num_start
        Ct_cumsum = cp.cumsum(Ct * time_frame_size[:, None], axis = 0)
        
        para_all = None
        errors_all = None
        for par_chunk in chunked_par_mat:
            M = generate_models(
                time_frame_size, Cr, Cr_cumsum, Ct, Ct_cumsum, Ti, par_chunk
            )
            para, errors = calculate_results(M, par_chunk, Ct, 
                                             par_chunk.shape[0], 
                                             thresh * par_chunk_num, write_paras, 
                                             model_0_prob_thres, vox_num_start, 
                                             distance_type = distance_type, 
                                             validation_mode = validation_mode, 
                                             hyperparameter = hyperparameter)
            if errors_all is None:
                para_all = para
                errors_all = errors
            else:
                para_all = cp.concatenate((para_all, para), axis = 1)
                errors_all = cp.concatenate((errors_all, errors), axis = 1)

        h = cp.quantile(errors_all, 1 / par_chunk_num, axis = 1)
        accepted_mask = errors_all <= h[:, None]
        accepted_size = int(cp.count_nonzero(accepted_mask[0]))
        par_mat_shape = para_all.shape
        accepted_mask = cp.broadcast_to(accepted_mask[:, :, None], 
                                        par_mat_shape)
        accepted = para_all[accepted_mask]
        accepted = accepted.reshape(par_mat_shape[0], accepted_size, par_mat_shape[-1])

        models = accepted[:, :, -1] ## (num_vox, accepted_size)
                                    ## array of models accepted for each voxel
        percentage_zeros = cp.mean(models == 0, axis = -1) ## along accepted_size axis
        models = (percentage_zeros < model_0_prob_thres).astype(cp.int32)
        model_p = cp.column_stack((voxel_numbers, models, percentage_zeros))
        model_p = model_p.get()

        if write_paras:
            accepted = accepted.reshape(par_mat_shape[0] * accepted_size, par_mat_shape[-1])
            voxel_numbers = voxel_numbers.repeat(accepted_size)
            accepted = cp.column_stack((voxel_numbers, accepted))
            accepted = accepted.get()
        else:
            accepted = None

        para_df, model_p_df = output_dataframe(accepted, model_p, write_paras)
        write_csv_chunks(para_df, model_p_df, path_output_para, path_output_model, 
                         write_paras, output_compressed)
        
        index += chunk_size

        # Stop the GPU timer
        gpu_end_event.record()
        gpu_end_event.synchronize()  # Ensure all GPU tasks are done before timing
        
        # Stop the CPU timers
        cpu_end_wall = time.time()
        cpu_end_process = os.times()

        # Calculate GPU time
        gpu_elapsed_time_ms = cp.cuda.get_elapsed_time(gpu_start_event, gpu_end_event)
        
        # Calculate CPU wall clock time (real time elapsed)
        cpu_elapsed_wall_time = cpu_end_wall - cpu_start_wall
        
        # Calculate total CPU time (user + system time) across all cores
        cpu_elapsed_process_time = (
            (cpu_end_process.user + cpu_end_process.system) -
            (cpu_start_process.user + cpu_start_process.system)
        )
        
        # Store the voxel processing time (both CPU and GPU)
        voxel_timing.append({
            "index": index - 3,
            "cpu_wall_clock_time_sec": cpu_elapsed_wall_time,
            "cpu_total_time_sec": cpu_elapsed_process_time,  # Total CPU time across all cores
            "gpu_time_ms": gpu_elapsed_time_ms
        })

    # Save the voxel processing times to a CSV file
    voxel_timing_df = pd.DataFrame(voxel_timing)
    voxel_timing_df.to_csv("voxel_processing_times.csv", index=False)

    if output_compressed:
        print("Compressing the model output...")
        compress_csv(path_output_model)
        print("Model output compressed...")

    print("vABC algorithm completed!")

def main():
    """
    Main function that executes the vABC algorithm.
    """

    """
    Input data file should have the shape:
    frame_length	Ti	        Cb	        0	        1	        2
    0	            0.133333333	0.632698	0.040760215	0.036601037	0.028408282
    0.166666667	    0.35	    140.5885	18.177458	17.420736	16.563547
    0.166666667	    0.516666667	17912.979	0	        0	        0
    0.166666667	    0.683333333	4444.976	8.17E-08	6.64E-08	4.08E-08
    0.166666667	    0.85	    2675.443	0.001335959	0.000509933	0.000515968

    The following columns are all TACs of the voxels.
    frame_length is the time frame size.
    Ti is the time after administration of the tracer.
    Cb is the whole blood input function.
        (Or equavalently Ca, the plasma input function)

    If input data is an HDF5 file, the key should be "df".
    """
    # path_data = "../Datasets_vPET-ABC/DigiMouse/Scale4/Voxel-wise-TACs-scale4-dif-time-vector_reformatted.csv"
    # path_data = "../Datasets_vPET-ABC/DigiMouse/Scale1/Voxel-wise-TACs-scale1-wholeBrain_reformatted.csv"
    # path_data = "../Datasets_vPET-ABC/DigiMouse/Scale1/Voxel-wise-TACs-scale1_reformatted.csv"
    # path_data = "vABC_data_activation.csv"
    path_data = "simulated_w_delays.csv"


    path_output_para = "parameters.csv"
    path_output_model = "model.csv"

    seed = 2024
    cp.random.seed(seed) ## for reproducibility

    chunk_size = 1 ## Adjust as needed, to prevent memory overflow

    S = 1*10**8 ## number of prior simulations
    thresh = 0.000001 ## threshold for acceptance
    model_0_prob_thres = 0.5 ## threshold for model 0 probability
    num_voxel = None ## number of voxels to process. If None, all voxels are
    write_paras = True ## flag indicating whether to write parameter posterior
    input_compressed = False ## flag indicating whether the input data is compressed (hdf5/csv)
    output_compressed = False
    ## flag indicating whether to compress the output 
    ## posteriors (hdf5/csv). Note that the model 
    ## probability posterior is always stored as a csv initially
    ## as it is relatively small, but can be compressed if needed.

    distance_type = "L1" ## distance function to use
    ## 
    validation_mode = False ## flag indicating whether to use validation mode

    par_mat = None
    if validation_mode:
    ## If validation mode is on, try to load precomputed models to save 
    ## computation time
        try:
            par_mat = cp.load("parameter_matrix.npz")["par_mat"]
            print("Priors loaded...")
        except FileNotFoundError:
            pass
    if par_mat is None:

        model = cp.random.binomial(1, 0.5, S)
        R1 = cp.random.uniform(0.2, 1.6, S)
        K2 = cp.random.uniform(0, 0.6, S)
        K2a = cp.random.uniform(0, 0.2, S)
        gamma = cp.random.uniform(0, 0.2, S)
        tD = cp.random.uniform(25, 35, S)
        tP = cp.random.uniform(tD + 1, 90, S)
        alpha = cp.random.uniform(0, 4, S)

        gamma[model == 0] = 0 ## model 0 for MRTM, model 1 for lp-nt

        par_mat = cp.column_stack((R1, K2, K2a, gamma, tD, tP, alpha, model))
        ## stacked as input
        cp.savez_compressed("parameter_matrix", par_mat = par_mat)
        print("Priors generated...")

    vABC(num_voxel, path_data, path_output_para, path_output_model, par_mat, 
         S, thresh, model_0_prob_thres, write_paras, input_compressed, 
         output_compressed, chunk_size, 
         distance_type = distance_type, validation_mode = validation_mode, 
         hyperparameter = None)
    
if __name__ == "__main__":
    print("Starting vABC algorithm...")
    main()