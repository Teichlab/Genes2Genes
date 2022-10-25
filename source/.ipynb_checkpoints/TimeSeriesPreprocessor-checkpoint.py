import numpy as np
import seaborn as sb
import torch

import MyFunctions
import MVG

class SummaryTimeSeries:
    
    def __init__(self, time_points, mean_trend, std_trend, data_bins, X,Y, cell_densities):
        self.time_points = np.asarray(time_points)
        self.mean_trend = np.asarray(mean_trend)
        self.std_trend = np.asarray(std_trend)
        self.data_bins = data_bins
        self.X = X
        self.Y = Y
        self.cell_densities = cell_densities
        self.intpl_means = None
        self.intpl_stds = None
        
    def plot_mean_trend(self, color='blue'):
        sb.lineplot(self.time_points, self.mean_trend, linewidth=3, color=color)
        
        
    def reverse_time_series(self):
        
        self.time_points = self.time_points[::-1]
        self.mean_trend = self.mean_trend[::-1]
        self.std_trend = self.std_trend[::-1]
      #  self.data_bins = self.data_bins[::-1]
        self.X = self.X [::-1]
        self.Y = self.Y[::-1]
        self.cell_densities = self.cell_densities[::-1]
        self.intpl_means = self.intpl_means[::-1]
        self.intpl_stds = self.intpl_stds[::-1]
        
        
class Prepocessor:
    
    def __init__(self, *args):
        if len(args)>1:
            GEX_MAT =args[0] 
            pseudotime_series = args[1]
            m = args[2]
            self.GEX_MAT = GEX_MAT
            self.pseudotime_series = pseudotime_series
            self.compute_cell_density_trend(m=m) 
        else:
            self.GEX_MAT = None
            self.pseudotime_series = None
            self.cell_densities = None
        
    
    def create_summary_trends(self, X, Y):

        # remember we have 100 synthetic cells per each time point
        mean_trend = []
        std_trend = [] 
        data_bins = []
        for t in range(len(self.artificial_time_points)):
            data_points = Y[X== self.artificial_time_points[t]]
            mean_trend.append(np.mean(data_points) ) 
            std_trend.append(np.std(data_points) )
            data_bins.append(data_points)

        return SummaryTimeSeries(self.artificial_time_points, mean_trend, std_trend, data_bins,X,Y,self.cell_densities)


    # data = dataframe, pseudotime series = array of pseudotimes for cells
    def create_equal_len_time_bins(self, gene, N_BINS = 10): 
        bins_indices = np.linspace(np.min(self.pseudotime_series), np.max(self.pseudotime_series), N_BINS+1 ) # bin margins
        bins_indices[N_BINS] = np.max(self.pseudotime_series) + 0.00001 # small jitter added to consistently mark the bin boundaries as [), [), .... ]]
        data_bins = []
        time_bins = []
        bin_compositions = []

        for i in range(len(bins_indices)):
            if(i==len(bins_indices)-1):
                break
            t = np.logical_and(self.pseudotime_series >= bins_indices[i], self.pseudotime_series < bins_indices[i+1])
            data_bins = data_bins + list(self.GEX_MAT.loc[t,gene])
            bin_compositions.append(len(self.GEX_MAT.loc[t,gene]))
            time_bins = time_bins + list(np.repeat(bins_indices[i+1], len(self.GEX_MAT[t]))) 
        return bins_indices[1:len(bins_indices)], time_bins, np.asarray(data_bins), bin_compositions



    # **** CellAlign paper's interpolation method based on Gaussian Kernel
    def interpolate_time_series(self, gene): # WINDOW_SIZE = 0.1 # default value used in CellAlign 

        intpl_gex = []
        for intpl_i in range(len(self.artificial_time_points)):
            weights = self.cell_weights[intpl_i]
            weighted_sum = 0.0
            for cell_i in range(len(self.pseudotime_series)):
                weighted_sum = weighted_sum + (weights[cell_i]*self.GEX_MAT[gene][cell_i]) 
            weighted_sum = weighted_sum/np.sum(weights)
            intpl_gex.append(weighted_sum)
        intpl_gex = np.asarray(intpl_gex).flatten() 
        #return intpl_gex, self.artificial_time_points

        # min max normalisation
        scaled_intpl_gex = []
        for i in range(len(intpl_gex)):
            scaled_intpl_gex.append((intpl_gex[i] - np.min(intpl_gex))/(np.max(intpl_gex) - np.min(intpl_gex) )) 
        return scaled_intpl_gex, self.artificial_time_points


    # My trial - interpolation of distributions based on Gaussian kernel (similar to above method but we get a distribution of artificial cells for interpolated time points now)
    #  weighted mean and weighted std based dist interpolation
    # Extending the CellAlign interpolation method based on Gaussian Kernel 
    def interpolate_time_series_distributions(self, gene, N=50, CONST_STD= False,WEIGHT_BY_CELL_DENSITY=False):

        torch.manual_seed(1)
        intpl_gex = []
        all_time_points = [] 
        intpl_means = []
        intpl_stds = []

        for intpl_i in range(len(self.artificial_time_points)):
            weights = self.cell_weights[intpl_i]
            weighted_sum = 0.0
            for cell_i in range(len(self.pseudotime_series)):
                weighted_sum = weighted_sum + (weights[cell_i]*self.GEX_MAT[gene][cell_i]) 
            weighted_sum = weighted_sum/np.sum(weights)
            dist_mean = weighted_sum

            if(CONST_STD): # for getting just the average trend across
                dist_std = 0.1
            else: # tweighted standard deviation 
                real_mean = np.mean(self.GEX_MAT[gene]) 
                weighted_sum_std = 0.0
                for cell_i in range(len(self.pseudotime_series)):
                    weighted_sum_std = weighted_sum_std + (weights[cell_i]*(( self.GEX_MAT[gene][cell_i] - real_mean) ** 2))
                n = len(self.pseudotime_series)
                weighted_std = np.sqrt(weighted_sum_std/(np.sum(weights) * (n-1)/n)) 
                #print(weighted_std)
                if(WEIGHT_BY_CELL_DENSITY):
                    weighted_std = weighted_std * self.cell_densities[intpl_i] # weighting according to cell density 
                dist_std = weighted_std
                #print(dist_std, ' -- ', self.cell_densities[intpl_i])
            if(dist_std==0 or np.isnan(dist_std)): # case of single data point or no data points
                dist_std = 0.1 #np.mean(summary_series_obj.std_trend)
            D,temp1,temp2 = MyFunctions.generate_random_dataset(N, dist_mean, dist_std)
            
            intpl_gex.append(D)
            intpl_means.append(dist_mean)
            intpl_stds.append(dist_std)
            all_time_points.append(np.repeat(self.artificial_time_points[intpl_i], N)) 
            
        return [np.asarray(intpl_gex).flatten(), np.asarray(all_time_points).flatten(), self.artificial_time_points, intpl_means, intpl_stds]
    

    def prepare_interpolated_gene_expression_series(self, gene, CONST_STD=False, WEIGHT_BY_CELL_DENSITY=False):
        # under the default setting: WINDOW_SIZE=0.1, N=50, m=50, CONST_STD=EVAL_AVERAGE_TREND
        # return list with intpl_ref, all_time_points, artificial_time_points, intpl_ref_means, intpl_ref_stds
       # intpl_out = self.interpolate_time_series_distributions(observed_data_mat, observed_data_time, gene)
        intpl_out = self.interpolate_time_series_distributions(gene, CONST_STD=CONST_STD, WEIGHT_BY_CELL_DENSITY= WEIGHT_BY_CELL_DENSITY)
        X = intpl_out[1]; Y =  intpl_out[0]; artificial_time_points = intpl_out[2]
        obj = self.create_summary_trends(X,Y) 
        obj.intpl_means = intpl_out[3]
        obj.intpl_stds = intpl_out[4]
        
        self.all_time_points = X # [TODO] -- repeats the same thing! To be done efficiently later!!!!
        
        return obj

    
    def compute_cell_density_trend(self, WINDOW_SIZE = 0.15, m=50):
        
        artificial_time_points = []
        for j in range(1,m):
            artificial_time_points.append((j-1)/(m-1))
        artificial_time_points = np.asarray(artificial_time_points)
        artificial_time_points = artificial_time_points[artificial_time_points > np.min(self.pseudotime_series)] 
        artificial_time_points = artificial_time_points[artificial_time_points < np.max(self.pseudotime_series)] 
            
        cell_densities = [] 
        cell_weights = {} 
        
        for intpl_i in range(len(artificial_time_points)):
            weights = []
            for cell_i in range(len(self.pseudotime_series)):
                w_i = np.exp(-((self.pseudotime_series[cell_i] - artificial_time_points[intpl_i])**2)/(WINDOW_SIZE**2)) 
                weights.append(w_i)
            # weighted cell density
            cell_densities.append(np.sum(weights))
            cell_weights[intpl_i] = np.asarray(weights) 
        cell_densities = np.asarray(cell_densities)
        cell_densities = cell_densities/len(self.pseudotime_series)
        
        # assigning second minimum in the case of 0 cell density values
        #second_min = np.sort(cell_densities)[1]  
        #for i in range(len(cell_densities)):
        #    if(cell_densities[i]==0):
        #        cell_densities[i]= second_min 
        
        self.cell_weights = cell_weights
        self.artificial_time_points = artificial_time_points
        
        self.cell_densities = cell_densities
        return cell_weights, artificial_time_points, cell_densities
    
        #scaled_cell_densities = Utils().minmax_normalise(cell_densities)
        # assigning second minimum in the case of 0 cell density values
        #second_min = np.sort(scaled_cell_densities)[1]  
        #for i in range(len(scaled_cell_densities)):
        #    if(scaled_cell_densities[i]==0):
        #        scaled_cell_densities[i]= second_min        
        #self.cell_densities = scaled_cell_densities
        #return cell_weights, artificial_time_points, scaled_cell_densities

# Later TODO: make superclass SummaryTimeSeries for both univariate and multivariate cases
class SummaryTimeSeriesMVG:
    
    def __init__(self, time_points, data_bins):
        self.time_points = np.asarray(time_points)
        self.data_bins = data_bins
        
        self.mean_trends = [] 
        for data_bin in data_bins:
            data_bin = torch.tensor(np.asarray(data_bin) )
            μ, C = MVG.compute_mml_estimates(data_bin, data_bin.shape[1], data_bin.shape[0]) 
            self.mean_trends.append(μ)
        


class Utils:
    
    def minmax_normalise(arr):
        
        norm_arr = []
        for i in range(len(arr)):
            norm_arr.append((arr[i] - np.min(arr))/(np.max(arr) - np.min(arr) )) 
        return norm_arr
    









# ATTIC CODE
            #print(weighted_std)
   #     if(summary_series_obj.time_points[0]>=artificial_time_points[intpl_i]):
   #         closest_bin_time_point_id = 0
   #     else:
   #         closest_bin_time_point_id = np.max(np.where(summary_series_obj.time_points <= artificial_time_points[intpl_i] ))+1
        
       # print('closest bin: ', closest_bin_time_point_id)
    #    if(closest_bin_time_point_id !=0):
    #        dist_var = (summary_series_obj.std_trend[closest_bin_time_point_id] + summary_series_obj.std_trend[closest_bin_time_point_id-1])/2.0
            #print(' mean : ',summary_series_obj.std_trend[closest_bin_time_point_id] ,  summary_series_obj.std_trend[closest_bin_time_point_id-1] )
     #   else:
     #       dist_var = summary_series_obj.std_trend[closest_bin_time_point_id]
        # assign variance of closest time bin cells of actual data
        
     #   dist_var = np.mean(summary_series_obj.std_trend) # assign mean variance across all cells
       # dist_var = 0.1
        
        
        
        
        
        