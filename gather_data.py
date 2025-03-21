import functions as funcs
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

output_dir = "/data/ERCblackholes4/aasnha2/for_aineias/plots3"
os.makedirs(output_dir, exist_ok=True)

snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableLowSNEff"
snap_dir2 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableLowSNEffHighRes"
snap_dir3 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
snap_dir4 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"

#cells with edge data
for snap_dir, snap_nums in [(snap_dir4, range(40,87))]: #40-87
    import os
    os.chdir("/data/ERCblackholes4/aasnha2/for_aineias/plots3")
    num_neighbours = 32
    # Initialize lists
    mass_flows = []
    neighbour_data_list = []
    edges_list = []
    global_data_list = []
    
    for snap_number in snap_nums:
        global_data, mass_flow, neighbour_data, edges = funcs.Fiacconi_mass_flux_edges(snap_dir, snap_number, num_neighbours=num_neighbours)
        mass_flows.append(mass_flow)
        neighbour_data_list.append(neighbour_data)
        edges_list.append(edges)
        global_data_list.append(global_data)
        print(f"Saved mass flow for snapshot {snap_number} for {os.path.basename(snap_dir)}")

    # Convert to numpy arrays
    mass_flows = np.array(mass_flows)
    neighbour_data_list = np.array(neighbour_data_list)
    global_data_list = np.array(global_data_list)

    # Save with edges as object array
    np.savez(f'untouched_mass_flow_edges_{num_neighbours}_{os.path.basename(snap_dir)}.npz', 
            mass_flow=mass_flows,
            neighbour_data=neighbour_data_list,
            global_data=global_data_list,
            allow_pickle=True)
    import pickle
    with open(f'edges_list_{num_neighbours}_{os.path.basename(snap_dir)}.pkl', 'wb') as f:
        pickle.dump(edges_list, f)

#cells without edge data
