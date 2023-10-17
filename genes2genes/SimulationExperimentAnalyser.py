# NEW ALIGNMENT ACCURACY STATISTIC CODE
import numpy as np
import regex as re
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import ClusterUtils

class SimulationExperimenter:
    
    def __init__(self, adata_ref, adata_query, aligner_all, CP_25, CP_05, CP_75, pattern_map): 
        self.adata_ref = adata_ref
        self.adata_query = adata_query
        self.aligner_all = aligner_all
        self.CP_25 = CP_25
        self.CP_05 = CP_05
        self.CP_75 = CP_75
        self.pattern_map = pattern_map

    def compute_match_statistics(self, existing_method_df, existing_method=True, print_stat=True):

        group_alignment_strings, gene_group = self.get_group_alignment_strings('AllMatch', existing_method, existing_method_df)
        #print(gene_group)
        n_false_mismatches = 0
        n_false_mismatched_alignments = 0
        alignment_state_count = 0 
        false_mismatch_counts = []
        for alignment_string in group_alignment_strings:
            mismatch_count = alignment_string.count('I') + alignment_string.count('D') 
            if(mismatch_count>0):
                n_false_mismatched_alignments+=1
                n_false_mismatches += mismatch_count
                false_mismatch_counts.append(mismatch_count)
            alignment_state_count += len(alignment_string)

        if(print_stat):
            print('Number of false mismatched alignments ', n_false_mismatched_alignments, ' = ',n_false_mismatched_alignments*100 / len(group_alignment_strings), '%' )
            print('Number of false mismatches ', n_false_mismatches, ' = ', n_false_mismatches*100  / alignment_state_count, '%' )
            print('mean false mismatch count for an alignment = ', np.mean(false_mismatch_counts))
            print('*****')
        return n_false_mismatched_alignments*100 / len(group_alignment_strings)

    def get_group_alignment_strings(self, pattern, existing_method=False,  existing_method_df = None):

        gene_group = list(self.adata_ref.var_names[self.adata_ref.var.gene_pattern == pattern]) 
        #print(gene_group)
        if(existing_method):
            gene_group  =np.intersect1d(list(existing_method_df.index), gene_group)
            df = existing_method_df.loc[gene_group]
            group_alignment_strings = [] 
            for i in range(df.shape[0]):
                group_alignment_strings.append(df['alignment_string'][i])
            group_alignment_strings = [a.replace(' ','')  for a in group_alignment_strings] 
        else:
            group_alignment_strings = []
            for g in gene_group:
                group_alignment_strings.append(self.aligner_all.results_map[g].alignment_str)

        return group_alignment_strings, gene_group


    def get_accuracy_stat_divergence(self, alignment_str,  divergence_mode = True):

        if(not divergence_mode):
            alignment_str = alignment_str[::-1]

        expected_pattern = '^[M/W/V]+[I/D]+$'
        swapped_pattern = '^[I/D]+[M/W/V]+$'

        false_start_mismatch_len = 0
        false_end_match_len = 0
        n_matches = 0
        n_false_intermediate_mismatches = 0
        end_mismatch_len = 0
        status = ''

        if(alignment_str.count('M') +  alignment_str.count('W')  + alignment_str.count('V') == 0):
            status = 'complete_mismatch'
            false_start_mismatch_len = -1
            false_end_match_len = -1
            n_false_intermediate_mismatches = -1
            n_matches = -1

        elif(alignment_str.count('I') +  alignment_str.count('D')== 0 ):
            status = 'complete_match'
            false_start_mismatch_len = -1
            false_end_match_len = -1
            n_false_intermediate_mismatches = -1
            n_matches = -1
        else:
            res = re.findall(expected_pattern, alignment_str)
            res_alt = re.findall(swapped_pattern, alignment_str)
            if(len(res)==1):
                status = 'expected_pattern' 
                n_matches = alignment_str.count('M') + alignment_str.count('W') + alignment_str.count('V')
                end_mismatch_len =  alignment_str.count('I') + alignment_str.count('D') 
            elif(len(res_alt)==1):
                status = 'swapped_pattern'
                false_start_mismatch_len = -1
                false_end_match_len = -1
                n_false_intermediate_mismatches = -1
                n_matches = -1
                end_mismatch_len = -1
            else:
                status = 'complex_pattern' 

                # check for false start mismatches
                false_start_mismatch_len = 0
                c=0
                while(alignment_str[c] in ['I','D']):
                    false_start_mismatch_len+=1
                    c+=1

                false_end_match_len = 0
                c=len(alignment_str)-1
                while(alignment_str[c] in ['M','W','V']):
                    false_end_match_len +=1 
                    c-=1

                # find intermediate number of false mismatches within matched region 
                # by first extracting the region between the first match and the last match
                match_regions = []
                for m in re.finditer('[M/V/W]+', alignment_str):
                    if(m.start(0) != m.end(0)):
                        match_regions.append([m.start(0), m.end(0)-1]) 

                if(false_end_match_len==0):
                    first_match_region = match_regions[0]
                    last_match_region = match_regions[len(match_regions)-1]
                else:
                    first_match_region = match_regions[0]
                    last_match_region = match_regions[len(match_regions)-2]

                intermediate_str = alignment_str[first_match_region[0]: last_match_region[1]+1] 

                n_matches = intermediate_str.count('M') + intermediate_str.count('W') + intermediate_str.count('V')
                n_false_intermediate_mismatches = intermediate_str.count('I') + intermediate_str.count('D')


                indel_regions = []
                for m in re.finditer('[I/D]+', alignment_str):
                    if(m.start(0) != m.end(0)):
                        indel_regions.append([m.start(0), m.end(0)-1]) 
                last_indel_region = indel_regions[len(indel_regions)-1]
                end_mismatch_len = len(alignment_str[last_indel_region[0]:last_indel_region[1]+1]) 

                # main statistics 
                #print('False start mismatch len: ', false_start_mismatch_len)
                #print('False end match len: ', false_end_match_len)
                #print('[start] Match length', n_matches)
                #print('end mismatch length')
                #print('# of false intermediate mismatches', n_false_intermediate_mismatches)
                #print('End mismatch end', end_mismatch_len)

        return status, false_start_mismatch_len, false_end_match_len, n_matches, n_false_intermediate_mismatches, end_mismatch_len


    def plot_validation_stat(self, accuracy_results, n_bins = 15, divergence=True):

       # plt.subplots(1,3, figsize=(15,3))
       # plt.subplot(1,3,1)
       # sb.heatmap(CP_25, square=True, cmap='jet') 
       # plt.subplot(1,3,2)
       # sb.heatmap(CP_05, square=True, cmap='jet') 
       # plt.subplot(1,3,3)
       # sb.heatmap(CP_75, square=True, cmap='jet') 

        #plt.savefig('changepoint_kernels.pdf')


       # plt.subplots(1,3, figsize=(15,3))
       # plt.subplot(1,3,1)
        a = pd.DataFrame(self.CP_25 > 0.01) 
       # sb.heatmap(a, square=True) 
        approx_bifurcation_start_point_25 = np.min(np.where(a.iloc[299]==True))
        #approx_bifurcation_start_point_25  = np.round((0.5* approx_bifurcation_start_point_25/150) ,2)
        approx_bifurcation_start_point_25  = np.round((approx_bifurcation_start_point_25/300) ,2)


     #   plt.subplot(1,3,2)
        a = pd.DataFrame(self.CP_05 > 0.01) 
       # sb.heatmap(a, square=True) 
        approx_bifurcation_start_point_05 = np.min(np.where(a.iloc[299]==True)) 
        #approx_bifurcation_start_point_05 = np.round((0.5* approx_bifurcation_start_point_05/150),2)
        approx_bifurcation_start_point_05 = np.round((approx_bifurcation_start_point_05/300),2)


      #  plt.subplot(1,3,3)
        a = pd.DataFrame(self.CP_75 > 0.01) 
       # sb.heatmap(a, square=True) 
        approx_bifurcation_start_point_75 = np.min(np.where(a.iloc[299]==True))
        #approx_bifurcation_start_point_75  = np.round((0.5* approx_bifurcation_start_point_75/150),2)
        approx_bifurcation_start_point_75  = np.round((approx_bifurcation_start_point_75/300),2)

        if(divergence):
            expected_len_25 = [n_bins*approx_bifurcation_start_point_25, n_bins*0.25]
            expected_len_05 = [n_bins*approx_bifurcation_start_point_05, n_bins*0.5]
            expected_len_75 = [n_bins*approx_bifurcation_start_point_75, n_bins*0.75]

            expected_mismatch_len_25 = [n_bins*(1-0.25),n_bins*(1-approx_bifurcation_start_point_25)]
            expected_mismatch_len_05 = [n_bins*(1-0.5), n_bins*(1-approx_bifurcation_start_point_05)]
            expected_mismatch_len_75 = [n_bins*(1-0.75),n_bins*(1-approx_bifurcation_start_point_75)]

            y1 = 'start_match_len'
            y2 = 'end_mismatch_len'
            y3 = 'false_start_mismatch_len'
            y4 = 'n_false_intermediate_mismatches'

            y1_ = 'Start match length'
            y2_ = 'End mismatch length'
            y3_ = 'False start mismatch length'
            y4_ = 'Number of false intermediate mismatches'

            print('Approx. bifurcation start i for cp=0.25 = ', approx_bifurcation_start_point_25 )
            print('Approx. bifurcation start i for cp=0.5 = ', approx_bifurcation_start_point_05 )
            print('Approx. bifurcation start i for cp=0.75 = ', approx_bifurcation_start_point_75 )

                # divegence
            df_025 = accuracy_results['Divergence_025']
            df_05 = accuracy_results['Divergence_05']
            df_075 = accuracy_results['Divergence_075']
        else:
            expected_len_25 = [n_bins*(approx_bifurcation_start_point_75),n_bins*0.75]
            expected_len_75 = [n_bins*(approx_bifurcation_start_point_25),n_bins*0.25]
            expected_len_05 = [n_bins*(approx_bifurcation_start_point_05),n_bins*0.5]

            expected_mismatch_len_25 = [n_bins*(1-0.75),n_bins*(1-approx_bifurcation_start_point_75)]
            expected_mismatch_len_75 = [n_bins*(1-0.25), n_bins*(1-approx_bifurcation_start_point_25)]
            expected_mismatch_len_05 = [n_bins*(1-0.5), n_bins*(1-approx_bifurcation_start_point_05)]

            y1='end_match_len'
            y2= 'start_mismatch_len'
            y3='false_end_mismatch_len'
            y4 = 'n_false_intermediate_mismatches'

            y1_='End match length'
            y2_= 'Start mismatch length'
            y3_='False end mismatch length'
            y4_ = 'Number of false intermediate mismatches'

            print('Approx. convergent start i for cp=0.25 = ', 1-approx_bifurcation_start_point_75 )
            print('Approx. convergent start i for cp=0.5 = ', 1-approx_bifurcation_start_point_05 )
            print('Approx. convergent start i for cp=0.75 = ', 1-approx_bifurcation_start_point_25 )

            # divegence
            df_025 = accuracy_results['Convergence_025']
            df_05 = accuracy_results['Convergence_05']
            df_075 = accuracy_results['Convergence_075']

        print('Expected match len for cp=0.25 = ', expected_len_25 )
        print('Expected match len for cp=0.5 = ', expected_len_05 )
        print('Expected match len  for cp=0.75 = ', expected_len_75 )

        print('Expected mismatch len for cp=0.25 = ', expected_mismatch_len_25 )
        print('Expected mismatch len for cp=0.5 = ', expected_mismatch_len_05 )
        print('Expected mismatch len  for cp=0.75 = ', expected_mismatch_len_75 )

        df_075['BF_approx'] = np.repeat('0.75', len(df_075))  
        df_025['BF_approx'] = np.repeat('0.25', len(df_025))  
        df_05['BF_approx'] = np.repeat('0.5', len(df_05)) 
        df = pd.concat( [df_025, df_05, df_075])
        df = df[df.status!='complete_mismatch']
        df = df[df.status!='swapped_pattern']
        df = df[df.status!='complete_match']
        
        # get the max mismatch length (across Is and Ds segments)
        mismatch_regions = []
        for a in df['alignment_str']:
            temp_reg = re.findall('[I/D]+',a)
            if(not divergence): 
                mismatch_regions.append(temp_reg[0]) # first mismatch region 
            else:
                mismatch_regions.append(temp_reg[len(temp_reg)-1]) # last mismatch region
        mismatch_lengths = [] 
        for a in mismatch_regions:
             mismatch_lengths.append(np.max([a.count('I'),a.count('D')]))
    
        if(divergence): # because the indel length will always be twice the expected length (# of Is == # of Ds)
            df['end_mismatch_len'] = mismatch_lengths
        else:
            df['start_mismatch_len'] = mismatch_lengths

        plt.subplots(1,4, figsize=(15,4))
        plt.subplot(1,4,1)
        g = sb.violinplot(data=df,  y = y1, x='BF_approx', cut=0)
        plt.xlabel('Approx bifurcation point')
        plt.title(y1_)
        plt.ylim([0,18])
        g.axhspan(expected_len_25[0], expected_len_25[1], xmin=0, xmax=0.35, alpha=0.2)
        g.axhspan(expected_len_05[0], expected_len_05[1], xmin=0.35, xmax=0.65,facecolor='orange', alpha=0.2)
        g.axhspan(expected_len_75[0], expected_len_75[1], xmin=0.65, xmax=1.0,facecolor='green', alpha=0.2)
        plt.ylabel(y1_)

        plt.subplot(1,4,2)
        g = sb.violinplot(data=df,  y = y2, x='BF_approx', cut=0)
        plt.xlabel('Approx bifurcation point')
        plt.ylabel('')
        plt.ylim([0,18])
        plt.title(y2_)
        g.axhspan(expected_mismatch_len_25[0], expected_mismatch_len_25[1], xmin=0, xmax=0.35,alpha=0.2)
        g.axhspan(expected_mismatch_len_05[0], expected_mismatch_len_05[1],xmin=0.35, xmax=0.65,facecolor='orange', alpha=0.2)
        g.axhspan(expected_mismatch_len_75[0], expected_mismatch_len_75[1],xmin=0.65, xmax=1.0,facecolor='green', alpha=0.2)
        plt.ylabel(y2_)

        plt.subplot(1,4,3)
        sb.violinplot(data=df,  y =  y3, x='BF_approx', cut=0)
        plt.xlabel('Approx bifurcation point')
        plt.ylabel('')
        plt.ylim([0,18])
        plt.title(y3_)
        plt.ylabel(y3_)

        plt.subplot(1,4,4)
        sb.violinplot(data=df,  y =  y4, x='BF_approx', cut=0)
        plt.xlabel('Approx bifurcation point')
        plt.ylabel('')
        plt.ylim([0,18])
        plt.title(y4_)
        plt.ylabel(y4_)

        plt.tight_layout()

        return df


    def compute_divergence_convergence_statistics(self, existing_method = False, tr_df = None, print_stat=True):

        divcov_alignment_accuracy_results = {}

        for PATTERN in [ 'Convergence_025', 'Convergence_05', 'Convergence_075','Divergence_025', 'Divergence_05', 'Divergence_075']:

            if(not existing_method): # G2G
                group_alignment_strings, gene_group = self.get_group_alignment_strings(PATTERN)
            else: # TrAGEDy
                group_alignment_strings, gene_group = self.get_group_alignment_strings(PATTERN, existing_method=True, existing_method_df=tr_df)

            accuracy_status = [] 
            for al in group_alignment_strings:
                status, false_start_mismatch_len, false_end_match_len, n_matches, n_false_intermediate_mismatches, end_mismatch_len  = self.get_accuracy_stat_divergence(al, divergence_mode=PATTERN.startswith('Div'))
                accuracy_status.append([status, false_start_mismatch_len, false_end_match_len, n_matches, n_false_intermediate_mismatches, end_mismatch_len])

            d = pd.DataFrame(accuracy_status)
            if(PATTERN.startswith('Div')):
                d.columns = ['status','false_start_mismatch_len','false_end_match_len','start_match_len','n_false_intermediate_mismatches','end_mismatch_len']
            else:
                d.columns = ['status','false_end_mismatch_len','false_start_match_len','end_match_len','n_false_intermediate_mismatches','start_mismatch_len']
            d['alignment_str'] = group_alignment_strings
            d['gene'] = gene_group
            if(print_stat):
                print(PATTERN, len(gene_group), np.unique(d['status'] , return_counts=True)) 

            divcov_alignment_accuracy_results[PATTERN] = d

        return divcov_alignment_accuracy_results 

    # clustering related

    def computeE(self, alignment_strings, metric):
        # compute distance matrix 
        print('compute distance matrix')
        dist_mat_functions = {'hamming': ClusterUtils.compute_hamming_dist_matrix, 'levenshtein': ClusterUtils.compute_levenshtein_dist_matrix}
        compute_dist_matrix = dist_mat_functions[metric]
        E = compute_dist_matrix(alignment_strings)
        return E

    def run_clustering(self, alignment_strings, metric, gene_names, DIST_THRESHOLD=0.2, experiment_mode=False, E=None):

        if(E is None):
            # compute distance matrix 
            E = self.computeE(alignment_strings, metric)

        if(experiment_mode):
            scores = [];  n_clusters = []; dist_thresholds = np.arange(0.01,1.0,0.01); score_modes = []; n_small_clusters = [] 
            eval_dists = []
            for D_THRESH in tqdm(dist_thresholds): 
                gene_clusters, cluster_ids, silhouette_score, silhouette_score_mode, n_small_cluster = ClusterUtils.run_agglomerative_clustering(E, gene_names, D_THRESH)

                if(len(gene_clusters.keys())==1):
                    break
                scores.append(silhouette_score)
                n_clusters.append(len(gene_clusters.keys()))
                score_modes.append(silhouette_score_mode)
                n_small_clusters.append(n_small_cluster)
                eval_dists.append(D_THRESH)

            plt.rcParams.update({'font.size': 14})
            plt.subplots(1,3,figsize=(10,5))
            plt.subplot(1,3,1)
            sb.lineplot(x=eval_dists, y=scores, color = 'blue', marker='o') 
            plt.xlabel('Distnace threshold')
            plt.ylabel('Mean Silhouette Score')
            plt.subplot(1,3,2)
            sb.lineplot(x=n_clusters, y=scores, color='red', marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Mean Silhouette Score')
            plt.subplot(1,3,3)
            sb.lineplot(x=eval_dists, y=n_clusters, color='green', marker='o')
            plt.xlabel('Distance threshold')
            plt.ylabel('Number of clusters')
            plt.tight_layout() 
            df = pd.DataFrame([eval_dists,scores,n_clusters]).transpose()
            df.columns = ['Distance threshold', 'Mean Silhouette Score','Number of clusters']
            return df

        else:
            print('run agglomerative clustering | ', np.round(DIST_THRESHOLD,2) )
            gene_clusters, cluster_ids, silhouette_score, silhouette_score_samples, n_small_cluster  = ClusterUtils.run_agglomerative_clustering(E, gene_names, DIST_THRESHOLD)
            print('silhouette_score: ', silhouette_score)
            return gene_clusters 

    def compute_misclustering_rate(self, gene_clusters, alignment_strings):
        misclustered_count = 0
        cid = 0
        for i in range(len(gene_clusters)):
            cluster = gene_clusters[i]
            cluster_pattern =[]
            for g in cluster:
                cluster_pattern.append(self.pattern_map[g])

            pattern_types = np.unique(cluster_pattern, return_counts=True)[0]
            pattern_counts = np.unique(cluster_pattern, return_counts=True)[1]

            if(len(pattern_types)>1):
                max_count = np.max(pattern_counts)
                # recording the number of outliers in a cl
                for c in pattern_counts:
                    if(c!=max_count):
                        misclustered_count += c
            #print(cid, pattern_types, pattern_counts, misclustered_count)#, ' || misclustered count = ',misclustered_count)
            cid+=1
        print('misclustered rate: ', misclustered_count*100/len(alignment_strings),'%')
        return misclustered_count*100/len(alignment_strings)

    def compute_cluster_diagnostics(self, alignment_strings, gene_names, distance_metric = 'levenshtein'):

        E = self.computeE(alignment_strings, metric=distance_metric)
        df = self.run_clustering(alignment_strings, metric=distance_metric, gene_names=gene_names, experiment_mode=True, E=E) 

        print('computing misclustering rates for different distance thresholds') 
        misclustering_rates = []
        distance_thresholds = []
        dist_range = list(df['Distance threshold'])
        for dist_thresh in dist_range: 
            gene_clusters = self.run_clustering(alignment_strings, metric=distance_metric, gene_names=gene_names, DIST_THRESHOLD=dist_thresh , experiment_mode=False, E=E) 
            mc = self.compute_misclustering_rate(gene_clusters, alignment_strings) 
            misclustering_rates.append(mc)
            distance_thresholds.append(dist_thresh)
        df['misclustering_rate']= misclustering_rates

        return E, df
