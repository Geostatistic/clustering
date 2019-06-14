# -------------------------------------------------------------------------------
#
# Clustering
# ***********
#
# This SGeMS plugin ...
#
# AUTHOR: Roberto Mentzingen Rolo
#
# -------------------------------------------------------------------------------

import math

import sgems

import numpy as np
import math

import sklearn
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from sklearn.mixture import GMM

import matplotlib.pyplot as plt

import pandas as pd

# Shows every parameter of the plugin in the command pannel
def read_params(a, j=''):
    for i in a:
        if (type(a[i]) != type({'a': 1})):
            print j + "['" + str(i) + "']=" + str(a[i])
        else:
            read_params(a[i], j + "['" + str(i) + "']")


# function that code the samples based on users inputed cutoffs
def sample_class_cutoff(prop, cutoffs):
    cutoffs = sorted(cutoffs)
    coded_samples = []
    for i in prop:
        if math.isnan(i):
            coded_samples.append(float('nan'))
        for j in range(len(cutoffs)):
            if j == 0:
                if i <= cutoffs[j]:
                    coded_samples.append(j + 1)
            if j == len(cutoffs) - 1:
                if i > cutoffs[j]:
                    coded_samples.append(j + 1)
            if 0 < j <= (len(cutoffs) - 1):
                if cutoffs[j - 1] < i <= cutoffs[j]:
                    coded_samples.append(j + 1)
    return coded_samples

# function that creates a isotopic subset in relation to a primary variable
def isotopic_dataset(grid, prim_var, sec_var):
    var_matrix = []

    var_matrix.append(np.array((sgems.get_property(grid, prim_var))))

    for i in sec_var:
        p = sgems.get_property(grid, i)
        var_matrix.append(np.array(p))

    var_matrix = np.array(var_matrix)

    #removing variables that are not isotopic in relation with the primary
    lst_mask = list()
    for i, ref in enumerate(var_matrix[0]):
        if math.isnan(ref):
            pass
        else:
            lst_mask.append(np.isnan(var_matrix[:, i]))

    lst_mask = np.array(lst_mask)

    mask_f = lst_mask.sum(axis=0).astype('bool')

    var_isotopic_matrix = var_matrix[~mask_f]

    print "You are using ",len(var_isotopic_matrix)," variables."

    nan_indices= []
    for i,j in enumerate(var_isotopic_matrix[0]):
        if math.isnan(j):
            nan_indices.append(i)

    var_isotopic_matrix_trans = var_isotopic_matrix.T

    var_isotopic_final = []
    for i in var_isotopic_matrix_trans:
        if not math.isnan(i[0]):
            var_isotopic_final.append(i)

    var_isotopic_final = np.array(var_isotopic_final)

    return var_isotopic_final, nan_indices

# variable creation function
def create_variable(grid, name, list):
    lst_props_grid = sgems.get_property_list(grid)
    prop_final_data_name = name

    if (prop_final_data_name in lst_props_grid):
        flag = 0
        i = 1
        while (flag == 0):
            test_name = prop_final_data_name + '-' + str(i)
            if (test_name not in lst_props_grid):
                flag = 1
                prop_final_data_name = test_name
            i = i + 1

    sgems.set_property(grid, prop_final_data_name, list)

class clustering:
    def __init__(self):
        pass

    def initialize(self, params):
        self.params = params
        return True

    def execute(self):

        #Execute the function read_params
        read_params(self.params)
        print self.params

        # ----------------------------------------------------------------------
        #
        # Cut-offs domaining
        #
        # ----------------------------------------------------------------------

        # checking if box is checked
        if self.params['cutoff_check_box']['value'] == str(1):

            # Getting variables
            prop = self.params['prop_cutoff']['property']
            grid_d = self.params['prop_cutoff']['grid']
            cutoffs_user = (self.params['cutoffs_user']['value']).split()
            prop_cutoff = sgems.get_property(grid_d, prop)

            # substituting commas for points in users inputed cutoffs
            cutoffs_user_no_comma = []
            for i in cutoffs_user:
                cutoffs_user_no_comma.append(float(i.replace(",", ".")))

            coded_dataset = sample_class_cutoff(prop_cutoff, cutoffs_user_no_comma)

            # setting the variable
            prop_final_data_name = 'coded_cutoff_'+self.params['prop_cutoff']['property']
            lst_props_grid = sgems.get_property_list(grid_d)

            if (prop_final_data_name in lst_props_grid):
                flag = 0
                i = 1
                while (flag == 0):
                    test_name = prop_final_data_name + '-' + str(i)
                    if (test_name not in lst_props_grid):
                        flag = 1
                        prop_final_data_name = test_name
                    i = i + 1

            sgems.set_property(grid_d, prop_final_data_name, coded_dataset)

        # ----------------------------------------------------------------------
        #
        # K-means clustering
        #
        # ----------------------------------------------------------------------

        # checking if box is checked
        if self.params['k_check_box']['value'] == str(1):

            # Getting variables
            grid_k = self.params['K_grid']['value']
            nclus = int(self.params['k_number']['value'])
            sec_props_k = (self.params['k_sec_var']['value']).split(';')
            prim_var_k = self.params['K_prim_var']['value']

            var_isotopic_kmeans, nan_indices = isotopic_dataset(grid_k, prim_var_k, sec_props_k)

            #runing kmeans
            k = KMeans(n_clusters=nclus).fit(var_isotopic_kmeans)
            RT = k.labels_

            RT_lst = []
            m=0
            for i in range(len(nan_indices)+len(RT)):
                check = True
                for j in nan_indices:
                    if i == j:
                        RT_lst.append(float('nan'))
                        check = False
                if check == True:
                    RT_lst.append(RT[m])
                    m = m+1

            create_variable(grid_k, 'KMeans', RT_lst)

        # ----------------------------------------------------------------------
        #
        # Hierarchical clustering
        #
        # ----------------------------------------------------------------------

        # checking if box is checked
        if self.params['hier_check_box']['value'] == str(1):

            # Getting variables
            grid_h = self.params['hier_grid']['value']
            sec_props_h = (self.params['hier_sec_var']['value']).split(';')
            prim_var_h = self.params['hier_prim_var']['value']
            criterion = self.params['criterion']['value']
            treshold = float(self.params['treshold']['value'])
            method_h = self.params['method']['value']
            dist_metric = self.params['dist_met']['value']

            var_isotopic_hier, nan_indices_h = isotopic_dataset(grid_h, prim_var_h, sec_props_h)

            # generate the linkage matrix
            Z = linkage(var_isotopic_hier, method = method_h, metric = dist_metric)

            df = pd.DataFrame(Z)
            df.to_csv('linkage_matrix.csv', index = False)

            plt.figure(figsize=(20,20))
            dn = dendrogram(Z)
            plt.savefig('dendogram.png')

            #printing cophenet corr.
            c, coph_dists = cophenet(Z, pdist(var_isotopic_hier))
            print "Cophenet correlation should be close to 1 : {}".format(c)

            #runnig hierarchical clustering
            try:
                RT_lst_h = fcluster(Z, treshold, criterion= criterion)
            except:
                print 'erro'

            RT_lst_h_final = []
            m = 0
            for i in range(len(nan_indices_h) + len(RT_lst_h)):
                check = True
                for j in nan_indices_h:
                    if i == j:
                        RT_lst_h_final.append(float('nan'))
                        check = False
                if check == True:
                    RT_lst_h_final.append(RT_lst_h[m])
                    m = m + 1

            create_variable(grid_h, 'Hierarchical', RT_lst_h_final)

        # ----------------------------------------------------------------------
        #
        # GMM clustering
        #
        # ----------------------------------------------------------------------

        # checking if box is checked
        if self.params['gmm_checkbox']['value'] == str(1):

            # getting variables
            grid_gmm = self.params['gmm_grid']['value']
            prim_var_gmm = self.params['gmm_prim']['value']
            sec_gmm = (self.params['gmm_sec']['value']).split(';')
            components = int(self.params['gmm_components']['value'])
            cov_type = self.params['cov_type']['value']

            var_isotopic_gmm, nan_indices_gmm = isotopic_dataset(grid_gmm, prim_var_gmm, sec_gmm)

            gmm = GMM(n_components = components, covariance_type= cov_type).fit(var_isotopic_gmm)

            RT_lst_gmm = gmm.predict(var_isotopic_gmm)

            probs = gmm.predict_proba(var_isotopic_gmm)
            probs_trans = probs.T

            RT_lst_gmm_final = []
            m = 0
            for i in range(len(nan_indices_gmm) + len(RT_lst_gmm)):
                check = True
                for j in nan_indices_gmm:
                    if i == j:
                        RT_lst_gmm_final.append(float('nan'))
                        check = False
                if check == True:
                    RT_lst_gmm_final.append(RT_lst_gmm[m])
                    m = m + 1

            create_variable(grid_gmm, 'GMM', RT_lst_gmm_final)

            for i,j in enumerate(probs_trans):
                RT_lst_gmm_probs = []
                m = 0
                for k in range(len(nan_indices_gmm) + len(RT_lst_gmm)):
                    check = True
                    for l in nan_indices_gmm:
                        if k == l:
                            RT_lst_gmm_probs.append(float('nan'))
                            check = False
                    if check == True:
                        RT_lst_gmm_probs.append(probs_trans[i][m])
                        m = m + 1

                create_variable(grid_gmm, 'GMM_prob_cluster_'+str(i), RT_lst_gmm_probs)

        return True

    def finalize(self):
        return True

    def name(self):
        return "clustering"

################################################################################
def get_plugins():
    return ["clustering"]
