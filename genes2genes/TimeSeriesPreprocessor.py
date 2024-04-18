import numpy as np
import seaborn as sb
import pandas as pd
import torch
import multiprocessing
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

from . import MyFunctions
from . import Utils

class TrajectoryInterpolator:
    
    """
    This class defines an interpolator function for a given gene expression time series, which prepares required summary statistics for interpolation
    """
    
    def __init__(self, adata, n_bins, adaptive_kernel = True, kernel_WINDOW_SIZE=0.1, raising_degree = 1):
        self.n_bins = n_bins
        self.adata = adata[np.argsort(adata.obs['time'])] 

        self.cell_pseudotimes = np.asarray(self.adata.obs.time)
        self.interpolation_points = np.linspace(0,1,n_bins) 
        self.kernel_WINDOW_SIZE = kernel_WINDOW_SIZE
        self.adaptive_kernel = adaptive_kernel
        self.k = raising_degree # the degree of stretch imposed for the window sizes from baseline kernel_WINDOW_SIZE  = 0.1
        
        self.mat = csr_matrix(self.adata.X.todense().transpose())
        self.N_PROCESSES = multiprocessing.cpu_count() 
        self.gene_list = self.adata.var_names
        
    def run(self):
        #print('computing absolute time diffs')
        self.abs_timediff_mat = self.compute_abs_timediff_mat()
        if(self.adaptive_kernel):
            #print('Running in adaptive interpolation mode')
            #print('computing an cell densities for adaptive interpolation')
            self.reciprocal_cell_density_estimates = self.compute_cell_densities()
            #print('computing adaptive win denomes')
            self.adaptive_win_denoms =  self.compute_adaptive_window_denominator()
        #print('computing cell weight matrix')
        self.cell_weight_mat = self.compute_Weight_matrix() 
        
    def compute_abs_timediff_mat(self): # interpolation time points x cells matrix
        df = [] 
        for i in self.interpolation_points:
            # absolute difference between actual pseudotime point of a cell and interpolation time point (needed to compute gaussian kernel later on)
            abs_dist = np.abs(np.asarray(self.cell_pseudotimes) - i) #np.repeat(i,len(self.cell_pseudotimes)) 
            df.append(abs_dist)
        df = pd.DataFrame(df); df.columns = self.adata.obs_names; df.index = self.interpolation_points
        return df
        
    def compute_cell_densities(self): # cell density vector across interpolation time points
        # compute cell density estimate for each interpolation point
        cell_density_estimates = [] 
        interpolation_points = self.interpolation_points
        cell_pseudotimes = self.cell_pseudotimes
        range_length_mid = interpolation_points[2] - interpolation_points[0] # constant across 
        range_length_corner = interpolation_points[1] - interpolation_points[0] # constant across 
        for i in range(len(interpolation_points)):
            prime_point = interpolation_points[i] 
            cell_density = 0.0 # per discrete point cell density = # of cells falling within interpolation time points [i-1,i+1] range window / window length
            if(i==0):
                logic = cell_pseudotimes <= interpolation_points[i+1]; range_length = range_length_corner
            elif(i==len(interpolation_points)-1):
                logic = cell_pseudotimes >= interpolation_points[i-1]; range_length = range_length_corner
            else:
                logic = np.logical_and(cell_pseudotimes <= interpolation_points[i+1], cell_pseudotimes >= interpolation_points[i-1])
                range_length = range_length_mid

            density_stat = np.count_nonzero(logic)
            density_stat = density_stat/range_length
            cell_density_estimates.append(density_stat)
        #print('** per unit cell density: ', cell_density_estimates)
        self.cell_density_estimates = cell_density_estimates
        cell_density_estimates  = [1/x for x in cell_density_estimates] # taking reciprocal for weighing
        
        #print('reciprocals: ', cell_density_estimates)
        # if this has inf values, use the max weight for them (otherwise it becomes inf resulting same weights 1.0 for all cells)
        arr = cell_density_estimates
        if(np.any(np.isinf(arr))):
            max_w = max(np.asarray(arr)[np.isfinite(arr)] ) 
            cell_density_estimates = np.where(np.isinf(arr), max_w, arr)
        #print('** adaptive weights -- ', cell_density_estimates)
        
        return cell_density_estimates
        
    def compute_adaptive_window_denominator(self): # for each interpolation time point

        cell_density_adaptive_weights  = self.reciprocal_cell_density_estimates

        # using min-max to stretch the range (for highly adapted window sizes having high window sizes)
        cell_density_adaptive_weights =np.asarray(cell_density_adaptive_weights) 
        scaler = MinMaxScaler()
        cell_density_adaptive_weights = scaler.fit_transform(cell_density_adaptive_weights.reshape(-1, 1)).flatten()
        cell_density_adaptive_weights = cell_density_adaptive_weights * self.k
        
        # ======= enforcing the same window_size = kernel_WINDOW_SIZE for the interpolation with the least weighted kernel window size
        adaptive_window_sizes = []
        for cd in cell_density_adaptive_weights:
            adaptive_window_sizes.append(cd*self.kernel_WINDOW_SIZE)  #weighing stadard window size
            
        # find the interpolation point for which the window_size weighted to be lowest -- furthest to kernel_WINDOW_SIZE
        temp = list(np.abs(adaptive_window_sizes - np.repeat(self.kernel_WINDOW_SIZE,self.n_bins)))
        least_affected_interpolation_point = temp.index(max(temp))
        residue = np.abs(self.kernel_WINDOW_SIZE - adaptive_window_sizes[least_affected_interpolation_point])
        if(self.k>1): # linear scaling to stretch the range of window size from 0.1 base line. 
            adaptive_window_sizes = adaptive_window_sizes + (residue/(self.k-1)) 
        else: 
            adaptive_window_sizes = adaptive_window_sizes + residue 
  
        # compute adaptive window size based denominator of Gaussian kernel for each cell for each interpolation time point
        W = [] 
        for adw in adaptive_window_sizes:
            adaptive_W_size = adw**2
            W.append(adaptive_W_size)
        self.adaptive_window_sizes = adaptive_window_sizes
            
        return W
        
    # compute Gaussian weights for each interpolation time point and cell
    def compute_Weight_matrix(self):
        if(self.adaptive_kernel):
            adaptive_win_denoms_mat = np.asarray([np.repeat(a, len(self.cell_pseudotimes)) for a in self.adaptive_win_denoms])
            W_matrix = pd.DataFrame(np.exp(-np.divide(np.array(self.abs_timediff_mat**2), adaptive_win_denoms_mat)))
        else:
            W_matrix = pd.DataFrame(np.exp(-np.array(self.abs_timediff_mat**2)/self.kernel_WINDOW_SIZE**2))
        W_matrix.columns = self.adata.obs_names
        self._real_intpl = self.interpolation_points
        #self.interpolation_points = [np.round(i,2) for i in  self.interpolation_points]
        W_matrix.index = self.interpolation_points
        #sb.heatmap(W_matrix)
        return W_matrix
    
    def get_effective_cell_pseudotime_range(self, i, effective_weight_threshold):
        effective_weights = self.cell_weight_mat.loc[self.interpolation_points[i]]
        cell_names = np.asarray(effective_weights.index)
        effective_weights = np.asarray(effective_weights)
        cell_ids = np.where(effective_weights>effective_weight_threshold)[0]
        effective_cell_names = cell_names[cell_ids]
        effective_cell_pseudotimes = self.cell_pseudotimes[cell_ids]
        return effective_cell_pseudotimes

    # plotting highly effective cell_contribution regions for given interpolation points based on adaptive weighted gaussian kernel
    def plot_effective_regions_for_interpolation_points(self, intpointsIdx2plots, effective_weight_threshold=0.5, plot=True):
        
        cmap = sb.color_palette("viridis", as_cmap=True)
        self.n_effective_cells = []
        for i in intpointsIdx2plots:
            x = self.get_effective_cell_pseudotime_range(i, effective_weight_threshold= effective_weight_threshold)
            self.n_effective_cells.append(len(x))
            if(plot):
                sb.kdeplot(x, fill=True, color=cmap(i/self.n_bins), clip=(0.0,1.0))

    
"""
The below functions define interpolation functions used by the above Interpolator object 
(defined outside class for time efficiency)
"""
# ====================== interpolation process of genes
def compute_stat(row, x, cell_densities, user_given_std):
            idx = row.name
            if(user_given_std[idx] < 0):
                cell_weights_sum = np.sum(row)

                # estimate weighted mean
                weighted_mean = np.dot(row, x)/cell_weights_sum
                #print(weighted_mean)

                # estimate weighted variance
                real_mean = np.mean(x); n = len(row)
                weighted_sum_std = np.dot(row, (x - real_mean) ** 2 )
                weighted_std = np.sqrt(weighted_sum_std/(cell_weights_sum * (n-1)/n)) 
                weighted_std = weighted_std * cell_densities[idx] # weighting according to cell density 
            else:
                weighted_mean = 0.0 
                weighted_std =  user_given_std[idx] #
            
            D,_,_ = MyFunctions.generate_random_dataset(50, weighted_mean, weighted_std)
            return np.asarray([weighted_mean, weighted_std, D], dtype=list)

#row = list(trajInterpolator.cell_weight_mat.loc[intpl_i])
def interpolate_gene_v2(i, trajInterpolator, user_given_std):
        torch.manual_seed(1)
        GENE = trajInterpolator.gene_list[i]
        #print(GENE)
        x = Utils.csr_mat_col_densify(trajInterpolator.mat ,i)
        N_cells= len(trajInterpolator.cell_pseudotimes)
        
        trajInterpolator.cell_weight_mat.index = range(0,len(trajInterpolator.cell_weight_mat))
        cell_densities = list(trajInterpolator.cell_weight_mat.apply(np.sum, axis=1)/N_cells)
        
        results = trajInterpolator.cell_weight_mat.apply(compute_stat, axis=1, args = ([x,cell_densities, user_given_std]), result_type='expand')
        results = pd.DataFrame(results)

        return SummaryTimeSeries(GENE, results[0], results[1], results[2], trajInterpolator.interpolation_points) 

class SummaryTimeSeries:
    """
    This class defines an interpolated time series object that carries the interpolated result of a gene expression time series
    """
    
    def __init__(self, gene_name, mean_trend, std_trend, intpl_gex, time_points):
        self.gene_name = gene_name
        self.mean_trend = np.asarray([np.mean(data_bin) for data_bin in intpl_gex]) # interpolated dist mean
        self.std_trend = np.asarray([np.std(data_bin) for data_bin in intpl_gex]) # interpolated dist std
        self.data_bins = list(intpl_gex)
        self.intpl_means = list(mean_trend) # actual weighted means
        self.intpl_stds = list(std_trend) # actual weighted stds
        self.time_points = np.asarray(time_points)
        
        self.Y = np.asarray([np.asarray(x) for x in self.data_bins]).flatten() 
        self.X = np.asarray([np.repeat(t,50) for t in self.time_points]).flatten() 
        
    def plot_mean_trend(self, color='midnightblue'):
        sb.lineplot(x= self.time_points, y=self.mean_trend, color=color, linewidth=4) 
        
    def plot_std_trend(self, color='midnightblue'):
        sb.lineplot(x= self.time_points, y=self.std_trend, color=color, linewidth=4) 
    
    