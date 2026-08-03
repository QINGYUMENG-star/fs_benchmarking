import numpy as np
import warnings  
warnings.filterwarnings("ignore")
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm

import time
################## Define basic functions ##################

def remove_constant_columns(X, epsilon=1e-10):
    """Remove columns with very small variance"""
    std_dev = np.std(X, axis=0)
    non_constant_cols = std_dev > epsilon
    return X[:, non_constant_cols], np.where(~non_constant_cols)[0]


def cap_extreme_values_numpy(data, lower_percentile=5, upper_percentile=95):
    """Cap extreme values in array"""
    lower_bound = np.percentile(data, lower_percentile, axis=0)
    upper_bound = np.percentile(data, upper_percentile, axis=0)
    return np.clip(data, lower_bound, upper_bound)


def balanced_sampling(Y, selected_samples_size, is_binary=True):
    """
    Perform balanced sampling for either binary or continuous data
    """
    if is_binary:
        # Binary classification sampling
        positive_indices = np.where(Y == 1)[0]
        negative_indices = np.where(Y == 0)[0]
        
        sampling_stats = {
            'total_positive': len(positive_indices),
            'total_negative': len(negative_indices),
            'selected_per_class': selected_samples_size // 2
        }
        
        if len(positive_indices) < selected_samples_size//2 or len(negative_indices) < selected_samples_size//2:
            raise ValueError(
                f"Insufficient samples. Positive: {len(positive_indices)}, "
                f"Negative: {len(negative_indices)}, "
                f"Required per class: {selected_samples_size//2}"
            )
        
        selected_positive = np.random.choice(positive_indices, size=selected_samples_size//2, replace=False)
        selected_negative = np.random.choice(negative_indices, size=selected_samples_size//2, replace=False)
        selected_indices = np.concatenate([selected_positive, selected_negative])
    
    else:
        # Continuous data sampling with stratification
        num_bins = 10
        bins = np.percentile(Y, np.linspace(0, 100, num_bins + 1))
        bin_indices = np.digitize(Y, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        samples_per_bin = selected_samples_size // num_bins
        remaining_samples = selected_samples_size % num_bins
        
        selected_indices = []
        sampling_stats = {'bin_counts': {}}
        
        for i in range(num_bins):
            bin_samples = np.where(bin_indices == i)[0]
            sampling_stats['bin_counts'][f'bin_{i}'] = len(bin_samples)
            
            if len(bin_samples) < samples_per_bin:
                selected_indices.extend(bin_samples)
                deficit = samples_per_bin - len(bin_samples)
                
                # Get samples from adjacent bins
                adjacent_bins = []
                if i > 0:
                    adjacent_bins.extend(np.where(bin_indices == i-1)[0])
                if i < num_bins - 1:
                    adjacent_bins.extend(np.where(bin_indices == i+1)[0])
                
                if adjacent_bins:
                    extra_samples = np.random.choice(
                        adjacent_bins,
                        size=min(deficit, len(adjacent_bins)),
                        replace=False
                    )
                    selected_indices.extend(extra_samples)
            else:
                selected = np.random.choice(bin_samples, size=samples_per_bin, replace=False)
                selected_indices.extend(selected)
        
        if remaining_samples > 0:
            remaining_indices = np.setdiff1d(np.arange(len(Y)), selected_indices)
            extra_samples = np.random.choice(
                remaining_indices,
                size=remaining_samples,
                replace=False
            )
            selected_indices.extend(extra_samples)
    
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    
    return selected_indices, sampling_stats

def process_features(data_selected):
    """Process and transform selected features"""
    # Split features into groups
    feature_groups = {
        'linear': data_selected[:, :5],
        'cos': data_selected[:, 5:8],
        'log': np.abs(data_selected[:, 8:11]),
        'power': data_selected[:, 11:14],
        'exp': data_selected[:, 14:17],
        'combine': data_selected[:, 17:20]
    }
    
    # Calculate transformations
    feature_sums = {
        'linear': np.sum(feature_groups['linear'], axis=1),
        'cos': np.cos(np.sum(feature_groups['cos'], axis=1)),
        'log': np.log(np.sum(feature_groups['log'], axis=1) + 1e-10),
        'power': (np.sum(feature_groups['power'], axis=1) ** 3),
        'exp': cap_extreme_values_numpy(np.exp(np.sum(feature_groups['exp'], axis=1))),
        'combine': cap_extreme_values_numpy(
            np.cos(np.sum(feature_groups['combine'], axis=1)) +
            np.log(np.sum(np.abs(feature_groups['combine']), axis=1) + 1e-10) +
            np.sum(feature_groups['combine'], axis=1) ** 3 +
            np.exp(np.sum(feature_groups['combine'], axis=1))
        )
    }
    
    return feature_sums

def generate_target(feature_sums, is_binary=True, noise_level=6):
    """Generate target variable based on features"""
    # Calculate weighted sum
    miu = (0.4 * feature_sums['linear'] + 
           5 * feature_sums['cos'] + 
           3 * feature_sums['log'] + 
           0.05 * feature_sums['power'] + 
           0.15 * feature_sums['exp'] + 
           0.05 * feature_sums['combine'])
    
    generation_stats = {}
    
    if is_binary:
        # Generate binary target
        probabilities = expit(miu - np.mean(miu))
        Y = np.random.binomial(1, probabilities)
        generation_stats.update({
            'mean_probability': np.mean(probabilities),
            'std_probability': np.std(probabilities)
        })
    else:
        # Generate continuous target
        err = np.random.normal(0, noise_level, len(miu))
        Y = miu + err
        Y = (Y - np.mean(Y)) / np.std(Y)  # Standardize
        generation_stats.update({
            'mean_target': np.mean(Y),
            'std_target': np.std(Y),
            'noise_level': noise_level
        })
    
    return Y, generation_stats


def process_iteration(gene_expression, selected_samples_size, num_selected_feature, 
                     is_binary=True, random_seed=42):
    """Process one iteration of data generation"""
    np.random.seed(random_seed)
    
    # Remove constant columns
    gene_expression, removed_cols = remove_constant_columns(gene_expression)

    # Select features
    selected_columns_indices = np.random.choice(gene_expression.shape[1], 
                                              size=num_selected_feature, 
                                              replace=False)
    data_selected = gene_expression[:, selected_columns_indices]

    # Process features
    feature_sums = process_features(data_selected)

    # Generate target variable
    Y, generation_stats = generate_target(feature_sums, is_binary)

    # Perform sampling
    selected_indices, sampling_stats = balanced_sampling(Y, selected_samples_size, is_binary)
    
    # Get final dataset
    balanced_X = gene_expression[selected_indices]
    balanced_Y = Y[selected_indices]
    balanced_data_selected = data_selected[selected_indices]
   
    # Combine statistics
    stats = {
        'generation_stats': generation_stats,
        'sampling_stats': sampling_stats
    }
    
    return balanced_X, balanced_Y, balanced_data_selected, selected_columns_indices, stats


def generate_data(gene_expression_path, output_path, selected_samples_size, 
                 num_selected_feature, is_binary=True, sims=10):
    """Main function to generate datasets"""
    # Load and preprocess data
    gene_expression_o = np.load(gene_expression_path)
    print("Original gene expression shape:", gene_expression_o.shape)
    
    # Standardize
    scaler = StandardScaler()
    gene_expression = scaler.fit_transform(gene_expression_o)
    
    # Create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Generate multiple datasets
    np.random.seed(12100126)
    
    for sim in tqdm(range(sims)):
        try:
            X, Y, data_selected, selected_features, stats = process_iteration(
                gene_expression, 
                selected_samples_size, 
                num_selected_feature, 
                is_binary,
                random_seed=sim
            )
            
            # Save data
            data_dict = {
                'X': X,
                'Y': Y,
                'selected_features': selected_features,
                'data_selected': data_selected,
                'stats': stats
            }

            np.savez(
                os.path.join(output_path, f'data_{sim}.npz'), 
                **data_dict
            )

        except ValueError as e:
            print(f"Error in simulation {sim + 1}: {str(e)}")
            continue

if __name__ == "__main__":
    # Set parameters
    params = {
        'gene_expression_path': './gene_expression.npy',
        'selected_samples_size': 5000,
        'num_selected_feature': 20,
        'is_binary': False,  # Set to False for continuous data
        'sims': 10
    }
    
    # Set output path based on data type
    data_type = 'binary' if params['is_binary'] else 'continuous'
    output_path = f"/nesi/nobackup/uoa04155/dimen_reduction/data/continuous/data_{data_type}_{params['selected_samples_size']}/"
    
    
    # Run main function
    generate_data(
        params['gene_expression_path'],
        output_path,
        params['selected_samples_size'],
        params['num_selected_feature'],
        params['is_binary'],
        params['sims']
    )
    
