#This code combines the data from low and high SNeff simulations into a single dataset. It also normalises this data.
import pickle
import functions as funcs
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
os.chdir("/data/ERCblackholes4/aasnha2/for_aineias/plots3")

#load data
HighSN = np.load("untouched_mass_flow_edges_32_NoBHFableHighSNEff.npz")
LowSN = np.load("untouched_mass_flow_edges_32_NoBHFableLowSNEff.npz")
'''HighSNRes = np.load("untouched_mass_flow_edges_32_NoBHFableHighSNEffHighRes.npz")
LowSNRes = np.load("untouched_mass_flow_edges_32_NoBHFableLowSNEffHighRes.npz")'''
#load edge data
with open("edges_list_32_NoBHFableHighSNEff.pkl", "rb") as f:
    HighSN_edges = pickle.load(f)
with open("edges_list_32_NoBHFableLowSNEff.pkl", "rb") as f:
    LowSN_edges = pickle.load(f)
'''with open("edges_list_32_NoBHFableHighSNEffHighRes.pkl", "rb") as f:
    HighSNRes_edges = pickle.load(f)
with open("edges_list_32_NoBHFableLowSNEffHighRes.pkl", "rb") as f:
    LowSNRes_edges = pickle.load(f)'''

#concatenate edge data
LowRes_edges = HighSN_edges + LowSN_edges
'''HighRes_edges = HighSNRes_edges + LowSNRes_edges'''
    
with open(f'LowRes_edges.pkl', 'wb') as f:
    pickle.dump(LowRes_edges, f)
'''with open(f'HighRes_edges.pkl', 'wb') as f:
    pickle.dump(HighRes_edges, f)'''

#load neighbour data
HighSN_neighbour_data = HighSN["neighbour_data"]
LowSN_neighbour_data = LowSN["neighbour_data"]
'''HighSNRes_neighbour_data = HighSNRes["neighbour_data"]
LowSNRes_neighbour_data = LowSNRes["neighbour_data"]'''

#concatenate neighbour data
LowRes_neighbour_data = np.concatenate((HighSN_neighbour_data, LowSN_neighbour_data), axis=0)
'''HighRes_neighbour_data = np.concatenate((HighSNRes_neighbour_data, LowSNRes_neighbour_data), axis=0)'''
LowRes_neighbour_data_norm = funcs.lognorm_data(LowRes_neighbour_data, log=False)
'''HighRes_neighbour_data_norm = funcs.lognorm_data(HighRes_neighbour_data, log=False)'''
np.save("LowRes_neighbour_data_norm.npy", LowRes_neighbour_data_norm)
'''np.save("HighRes_neighbour_data_norm.npy", HighRes_neighbour_data_norm)'''

#load mass flow data
HighSN_mass_flow = HighSN["mass_flow"]
LowSN_mass_flow = LowSN["mass_flow"]
'''HighSNRes_mass_flow = HighSNRes["mass_flow"]
LowSNRes_mass_flow = LowSNRes["mass_flow"]'''

#concatenate mass flow data
LowRes_mass_flow = np.concatenate((HighSN_mass_flow, LowSN_mass_flow), axis=0)
'''HighRes_mass_flow = np.concatenate((HighSNRes_mass_flow, LowSNRes_mass_flow), axis=0)'''
np.save("LowRes_mass_flow.npy", LowRes_mass_flow)
'''np.save("HighRes_mass_flow.npy", HighRes_mass_flow)'''

#load global data
HighSN_global_data = HighSN["global_data"]
LowSN_global_data = LowSN["global_data"]
'''HighSNRes_global_data = HighSNRes["global_data"]
LowSNRes_global_data = LowSNRes["global_data"]'''

#concatenate global data
LowRes_global_data = np.concatenate((HighSN_global_data, LowSN_global_data), axis=0)
'''HighRes_global_data = np.concatenate((HighSNRes_global_data, LowSNRes_global_data), axis=0)'''
np.save("LowRes_global_data.npy", LowRes_global_data)
'''np.save("HighRes_global_data.npy", HighRes_global_data)'''
