import regex as re
import OrgAlign as orgalign 
import numpy as np 
from tqdm import tqdm 
import pandas as pd
from tabulate import tabulate
from scipy.spatial import distance

class TempFSMObj:
    
    def __init__(self, al_str, gene):
        self.al_str = al_str
        self.fsm, self.counts = self.get_transition_probs(al_str)
        self.al_length = len(al_str)
        self.gene = gene
        
    def get_transition_probs(self, al_str):
        transition_counts =  {'MM':0,'MI':0, 'MD':0,'MW':0,'MV':0, 'II':0, 'IM':0,'ID':0, 'IW':0, 'IV':0,'DD':0, 'DM':0,'DI':0, 'DW':0, 'DV':0, 
                             'WM':0,'WW':0,'WV':0,'WI':0,'WV':0, 'VM':0,'VW':0,'VV':0,'VI':0,'VV':0}
        sum_transitions = 0
        for key in transition_counts.keys():
            transition_counts[key] = len(re.findall(key,al_str, overlapped=True)) + 1 # Adding pseudocount to avoid log(0)
            sum_transitions += transition_counts[key]

        transition_probs = transition_counts.copy()

        for key in transition_counts.keys():
            transition_probs[key] = transition_counts[key]/sum_transitions
        return transition_probs, transition_counts
    
def compute_msg_len(transition_counts, fsm, al_len):
    
    msg_len = 0.0
    for key in transition_counts.keys():
        msg_len = -np.log(fsm[key])*transition_counts[key]
    msg_len = msg_len/al_len
    return msg_len

def pairwise_alignment_dist_v2(a1,a2):
    
    x1 = compute_msg_len(a1.counts, a1.fsm, a1.al_length)
    y1 = compute_msg_len(a2.counts, a1.fsm, a2.al_length)
    x2 = compute_msg_len(a1.counts, a2.fsm, a1.al_length)
    y2 = compute_msg_len(a2.counts, a2.fsm, a2.al_length)

    return (np.abs(x1-y1) + np.abs(x2-y2))/2

def get_region_str(al_str):
        prev = ''
        i=0
        regions = ''
        for i in range(len(al_str)):
            if(i==0):
                regions += al_str[i]
                continue
            if(al_str[i-1]==al_str[i]):
                continue
            else:
                regions += al_str[i]
                continue
        return regions
    
def test_unique_index_sums(a):
    index_sum = 0
    m = {'M':0,'I':0,'D':0,'W':0,'V':0}

    l = 0
    for i in range(len(a)):
        #print('==== ', i,m )
        if(i==len(a)-1):
            if(a[i-1]==a[i]):
                m[a[i]] += (index_sum + i)/(l+1)
            else:
                m[a[i-1]] += index_sum/l
                index_sum = 0
                m[a[i]] += (index_sum + i)
            #print(m)
            #print(i,index_sum/l) 
            break
            
        if(i==0 or a[i-1]==a[i]):
            index_sum += i
            l+=1
        else:
            #print('^',m[a[i-1]],a[i-1])
            m[a[i-1]] = m[a[i-1]] +  (index_sum/l)
            #print('*',index_sum/l,i,m[a[i-1]], a[i-1])
            #print(m)
            index_sum = 0
            index_sum += i
            l=1
        #print(i,index_sum/l)    
        
    return m 

class AlignmentDist:
    
    def __init__(self, aligner_obj):
        self.alignments = aligner_obj.results
        self.gene_list = aligner_obj.gene_list
        self.results_map = aligner_obj.results_map
        self.results = aligner_obj.results
        
     # computing pairwise polygon based distance between each pair of alignments in the set of all gene ref-query alignments
    def compute_polygon_area_alignment_dist(self):
        
        DistMat = []
        for i in range(len(self.alignments)):
            DistMat.append(np.repeat(-1,len(self.alignments)))
        for i in tqdm(range(len(self.alignments))):
            for j in range(len(self.alignments)):
                if(DistMat[i][j] < 0):
                    DistMat[i][j] = orgalign.Utils().compute_alignment_area_diff_distance(self.alignments[i].alignment_str, self.alignments[j].alignment_str
                                                    ,self.alignments[i].fwd_DP.S_len, self.alignments[i].fwd_DP.T_len )
                else:
                    DistMat[i][j] = DistMat[j][i]
                # DistMat[i][j] = DistMat[i][j] + textdistance.identity.distance(alignments[i].alignment_str, alignments[j].alignment_str)
        DistMat = pd.DataFrame(DistMat)
        DistMat.index = self.gene_list
        DistMat.columns = self.gene_list
        
        DistMat/np.max(np.asarray(DistMat).flatten())
        
        return DistMat

    def compute_alignment_ensemble_distance_matrix(self, scheme):

        PolygonDistMat = self.compute_polygon_area_alignment_dist()
        if(scheme==1):
            return PolygonDistMat

        FSA_objects = []
        FSA_objects_regionwise = []

        for i in range(len(self.alignments)):
            FSA_objects.append( TempFSMObj(self.alignments[i].alignment_str,self.alignments[i].gene ) )
            region_str = get_region_str(self.alignments[i].alignment_str)
            FSA_objects_regionwise.append(TempFSMObj(region_str,self.alignments[i].gene ))
            self.alignments[i].unique_index_sums = list(test_unique_index_sums(self.alignments[i].alignment_str).values())
            self.alignments[i].region_str = region_str

        Mat = []; Mat_ui = []
        for i in range(len(self.alignments)):
            Mat.append(np.repeat(-1.0,len(self.alignments)))
            Mat_ui.append(np.repeat(-1.0,len(self.alignments)))

        for i in range(len(self.alignments)):
            for j in range(len(self.alignments)):
                if(i==j):
                    Mat[i][j] = 0.0; Mat_ui[i][j] = 0.0
                if(Mat[i][j]<0):
                    Mat[i][j] = pairwise_alignment_dist_v2(FSA_objects[i],FSA_objects[j])
                    Mat_ui[i][j] = distance.euclidean(self.alignments[i].unique_index_sums,self.alignments[j].unique_index_sums)

        LikelihoodDistMat = pd.DataFrame(Mat)
        LikelihoodDistMat.columns = self.gene_list
        LikelihoodDistMat.index = self.gene_list
        LikelihoodDistMat = (LikelihoodDistMat/np.max(np.max(LikelihoodDistMat )))
        IndexSumDistMat = pd.DataFrame(Mat_ui)
        IndexSumDistMat.columns = self.gene_list
        IndexSumDistMat.index = self.gene_list
        IndexSumDistMat = IndexSumDistMat /np.max(np.max(IndexSumDistMat))

        if(scheme==2):
            return LikelihoodDistMat 
        elif(scheme==3):
            return  IndexSumDistMat
        elif(scheme==0):
            joint_mat = PolygonDistMat + LikelihoodDistMat + IndexSumDistMat
            return joint_mat/3
        elif(scheme==4):
            joint_mat = PolygonDistMat + LikelihoodDistMat
            return joint_mat/2
        elif(scheme==5):
            joint_mat = LikelihoodDistMat  + IndexSumDistMat
            return joint_mat/2
        elif(scheme==6):
            joint_mat = PolygonDistMat  + IndexSumDistMat
            return joint_mat/2
        
    
    
    def order_genes_by_alignments(self):
    
        indices = []
        genes = []
        gene_strs = []
        first_lengths= [] 
        
        for a in self.results:
            gene_strs.append(a.alignment_str)
            genes.append(a.gene)
            w_index = a.alignment_str.find('W')
            m_index = a.alignment_str.find('M')
            v_index = a.alignment_str.find('V')
            if(w_index<0):
                w_index = np.inf
            if(m_index<0):
                m_index = np.inf
            if(v_index<0):
                v_index = np.inf

            if(m_index<0):
                if(w_index >=0 and (w_index<v_index)):
                    first_index = w_index
                elif(v_index >=0 and (v_index<w_index)):
                    first_index = v_index
                else:
                    first_index = -1
            else:
                if((m_index<w_index) and (m_index<v_index)):
                    first_index = m_index
                elif((w_index<m_index) and (w_index<v_index)):
                    first_index = w_index
                elif((v_index<m_index) and (v_index<w_index)):
                    first_index = v_index
                else:
                    first_index = -1
            indices.append(first_index)

            l = 1
            r = first_index
            while(True):
                if(r+1==len( a.alignment_str)):
                    break
                curr_state =  a.alignment_str[r]
                next_state =  a.alignment_str[r+1]
                #print('',curr_state, next_state)
                if(next_state in ['M','W','V']):
                    l+=1
                    r+=1
                else:
                    break

            first_lengths.append(l)

        df = pd.DataFrame([genes, gene_strs, indices, first_lengths]).transpose()
        df.columns = ['genes','alignment','start_m_index','first_lengths']
        df = df.sort_values(['start_m_index','first_lengths'], ascending=[True,False])

        sorted_gene_list = df.genes
        table = []
        for gene in sorted_gene_list :
            #print(gene,aligner.results_map[gene].colored_alignment_str)
            table.append([gene,self.results_map[gene].colored_alignment_str]) 

        print(tabulate(table, headers=['Gene','Alignment']))
        return sorted_gene_list
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    