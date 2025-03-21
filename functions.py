#This file contains all the functions that are important to this project

import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

#This calculates the mass flux at a given radius. You must give it a shell size to work out the mass flow.
def mass_flux_at_radius(snap_dir, snap_number, radius, delta_r):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    import cosmo_utils as cu
    """
    Calculate the mass flow rate at a given radius and delta_r
    """

    h = 0.679 #dimensionless Hubble constant

    #get host group
    host_groups = cu.get_host_groups(snap_dir, snap_number, 0)
    gal_inds = host_groups == 0

    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    subhalovel = rff.get_subhalo_data(fof_file, "SubhaloVel")[0] #peculiar velocity of the subhalo (km/s)
    
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True)[gal_inds] #use 0 because gas is particle type 0. Position of every gas particle in the snapshot (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True)[gal_inds] #mass of every gas particle in the snapshot (Msun)
    vel0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True)[gal_inds] #velocity of every gas particle in the snapshot (km/s)

    #centre subhalopos
    pos0_c = pos0 - subhalopos #recentre the position of every gas particle on the subhalo
    vel0_c = vel0 - subhalovel #recentre the velocity of every gas particle on the subhalo

    r = np.sqrt(np.sum(pos0_c**2, axis=1)) #distance of each particle from centre of subhalo (kpc)
    shell_indices = np.where((r >= radius - delta_r/2) & (r < radius + delta_r/2))[0] #indices of gas particles within delta_r of the virial radius

    #shell properties
    pos_shell = pos0_c[shell_indices]
    vel_shell = vel0_c[shell_indices]
    mass_shell = mass0[shell_indices]

    # Normalize the position vectors to get radial direction
    r_shell = np.sqrt(np.sum(pos_shell**2, axis=1))
    pos_shell_normalized = pos_shell / r_shell[:, np.newaxis]
    
    # Get radial velocity by projecting velocity onto normalized position vector
    v_r_shell = np.sum(vel_shell * pos_shell_normalized, axis=1)

    #Calculate net mass flow rate
    mdot_tot = np.sum(mass_shell*v_r_shell/delta_r) #units of Msun*km/s/kpc
    conversion_factor = 3.241028867350146e-17  # 1 km / 1 kpc
    seconds_per_year = 31557600
    mdot_tot = mdot_tot*conversion_factor*seconds_per_year #negative if inflowing. This is in Msun/yr

    return mdot_tot

#This is used in Koudmani_mass_flux
def spline_kernel(rs, h):
    import numpy as np
    Ws = np.zeros(len(rs))
    prefactor = 8/(np.pi*h**3)
    for i in range(len(rs)):
        if 0 <= rs[i]/h <= 1/2:
            Ws[i] = prefactor*(1 - 6*(rs[i]/h)**2 + 6*(rs[i]/h)**3)
        elif 1/2 < rs[i]/h <= 1:
            Ws[i] = prefactor*2*(1 - rs[i]/h)**3
        else:
            Ws[i] = 0
    return Ws

#This is the way to calculate mass flow onto a black hole, given by Fiacconi et al. 2018. It also gives the properties of the gas cells used to calculate the mass flow.
def Fiacconi_mass_flux(snap_dir, snap_number, num_neighbours=32):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    from scipy import spatial
    from scipy import constants
    import cosmo_utils as cu
    
    #constants
    h = 0.679 #dimensionless Hubble constant
    k_B = constants.k*1e7 #Boltzmann constant (erg/K)
    m_proton = constants.m_p*1e3 #proton mass (g)
    X_H = 0.76 #hydrogen mass fraction
    gamma = 5/3

    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)

    #get host group
    host_groups = cu.get_host_groups(snap_dir, snap_number, 0)
    gal_inds = host_groups == 0

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    subhalovel = rff.get_subhalo_data(fof_file, "SubhaloVel")[0] #peculiar velocity of the subhalo (km/s). This acts as the velocity of the central black hole.

    #Need position, velocity, and density of the gas particles
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True)[gal_inds] #gas particle positions (kpc)
    density0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Density"), a, h, density=True)[gal_inds] #density of every gas particle in the snapshot (Msun/kpc^3)
    v_gas0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True)[gal_inds] #velocity of every gas particle in the snapshot (km/s)
    volume0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Volume"), a, h, volume=True)[gal_inds] #volume of every gas particle in the snapshot (kpc^3)
    internal_energy0 = rsf.get_snap_data(snap_name,0,"InternalEnergy")[gal_inds] #internal energy of every gas particle in the snapshot (km/s)^2
    electron_abundance0 = rsf.get_snap_data(snap_name,0,"ElectronAbundance")[gal_inds] #electron abundance of every gas particle in the snapshot (dimensionless)
    neutral_hydrogen_abundance0 = rsf.get_snap_data(snap_name,0,"NeutralHydrogenAbundance")[gal_inds] #neutral hydrogen abundance of every gas particle in the snapshot (dimensionless)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True)[gal_inds] #mass of every gas particle in the snapshot (Msun)

    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0)

    #Find neighbour indices
    neighbour_distances, neighbour_inds = gas_tree.query(subhalopos, k=num_neighbours+1)
    smoothing_length = np.max(neighbour_distances) # smoothing length used

    #Neighbour variables (taking out the values for the central particle)
    Ws = spline_kernel(neighbour_distances, smoothing_length)[1:]
    neighbour_pos = pos0[neighbour_inds[1:]] - subhalopos
    neighbour_densities = density0[neighbour_inds[1:]]
    neighbour_velocities = v_gas0[neighbour_inds[1:]] - subhalovel
    neighbour_volumes = volume0[neighbour_inds[1:]]
    neighbour_internal_energy = (internal_energy0[neighbour_inds[1:]])*1e10 #internal energy of every gas particle in the snapshot (cm/s)^2
    neighbour_electron_abundance = electron_abundance0[neighbour_inds[1:]]
    neighbour_neutral_hydrogen_abundance = neutral_hydrogen_abundance0[neighbour_inds[1:]]
    neighbour_masses = mass0[neighbour_inds[1:]]

    #calculate temperature
    mu = 4*m_proton/(1+3*X_H+4*X_H*neighbour_electron_abundance)
    neighbour_T = (gamma-1)*neighbour_internal_energy*mu/k_B #temperature of gas cells (K)

    #calculate radial velocity
    neighbour_radial_velocities = np.sum(neighbour_pos*neighbour_velocities, axis=1)/np.sqrt(np.sum(neighbour_pos**2, axis=1))

    #Calculate the mass flux
    mass_flux = np.sum(Ws*neighbour_densities*neighbour_radial_velocities)/np.sum(Ws)

    #Calculate the area of the sphere
    r_inflow = np.sum((3*neighbour_volumes/(4*np.pi))**(1/3)*Ws)/np.sum(Ws)
    area = 4*np.pi*r_inflow**2

    conversion_factor = 3.241028867350146e-17  # 1 km / 1 kpc
    seconds_per_year = 31557600
    mass_flow = area*mass_flux*conversion_factor*seconds_per_year #negative if inflowing. This is in Msun/yr
    return mass_flow, np.vstack((neighbour_distances[1:], neighbour_radial_velocities, neighbour_densities, neighbour_volumes, neighbour_masses, neighbour_T, neighbour_neutral_hydrogen_abundance)).T #now each row is a variable (neighbour_data[0,:] are all the distances)

#log and normalise data. Nolog gives the coumns that shouldn't be logged (i.e. velocity as it has positive and negative values)
def lognorm_data(data, log=True, nolog = [1]):
    dimensions = data.shape
    num_snapshots = dimensions[0]
    num_neighbours = dimensions[1]
    num_variables = dimensions[2]
    data = data.astype(np.float64)
    # Apply log transform to all variables except those specified in nolog
    if log:
        for i in range(data.shape[2]):  # Loop through variables
            if i not in nolog:
                data[:, :, i] = np.log10(data[:, :, i])
    data = data.reshape((num_snapshots*num_neighbours, num_variables))
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    data = (data - means)/stds
    data = data.reshape((num_snapshots, num_neighbours, num_variables))
    return data #note that data is now in the form (snapshots, neighbours, variables) e.g. (125, 32, 4). 
                #To get information about the 1st neighbour of the first snapshot: data[0,0,:]

#This function applies a symmetric logarithm to data
def symlog_transform(data, linthresh):
    """
    Apply symmetric logarithmic (symlog) transformation to data.
    """
    pos_mask = data > linthresh
    neg_mask = data < -linthresh
    lin_mask = ~(pos_mask | neg_mask)
    transformed = np.zeros_like(data, dtype=float)
    transformed[pos_mask] = np.log10(data[pos_mask] / linthresh)
    transformed[neg_mask] = -np.log10(-data[neg_mask] / linthresh)
    transformed[lin_mask] = data[lin_mask] / linthresh
    return transformed

def find_edges(positions):
    from scipy.spatial import Voronoi
    import numpy as np

    # Create Voronoi tessellation from positions
    vor = Voronoi(positions)
    # Find maximum magnitude in positions
    max_magnitude = np.max(np.linalg.norm(positions, axis=1))
    
    # Initialize empty list to store edges
    edges = []
    
    # Loop through ridge points to find edges
    for ridge in vor.ridge_points:
        # Each ridge connects two points - add both directions for undirected graph
        edges.append([ridge[0], ridge[1]])
        edges.append([ridge[1], ridge[0]])
        
    # Convert to numpy array and transpose to get (2, num_edges) shape
    edges = np.array(edges).T
    return edges

#The following functions were used for data processing for the graph neural network. It gives you global properties, the mass flow rate, the properties of each cell (including 3D coordinates), and edges between cells.
def Fiacconi_mass_flux_edges(snap_dir, snap_number, num_neighbours=32):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    from scipy import spatial
    from scipy import constants
    import cosmo_utils as cu
    
    #constants
    h = 0.679 #dimensionless Hubble constant
    k_B = constants.k*1e7 #Boltzmann constant (erg/K)
    m_proton = constants.m_p*1e3 #proton mass (g)
    X_H = 0.76 #hydrogen mass fraction
    gamma = 5/3

    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)

    #get host group
    host_groups = cu.get_host_groups(snap_dir, snap_number, 0)
    gal_inds = host_groups == 0

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    subhalovel = rff.get_subhalo_data(fof_file, "SubhaloVel")[0] #peculiar velocity of the subhalo (km/s). This acts as the velocity of the central black hole.
    redshift = rff.get_attribute(fof_file, "Redshift")
    subhalomass = rff.get_subhalo_data(fof_file, 'SubhaloMass')[0]

    #Need position, velocity, and density of the gas particles
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True)[gal_inds] #gas particle positions (kpc)
    density0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Density"), a, h, density=True)[gal_inds] #density of every gas particle in the snapshot (Msun/kpc^3)
    v_gas0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True)[gal_inds] #velocity of every gas particle in the snapshot (km/s)
    volume0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Volume"), a, h, volume=True)[gal_inds] #volume of every gas particle in the snapshot (kpc^3)
    internal_energy0 = (rsf.get_snap_data(snap_name,0,"InternalEnergy")[gal_inds])*1e10 #internal energy of every gas particle in the snapshot (cm/s)^2
    electron_abundance0 = rsf.get_snap_data(snap_name,0,"ElectronAbundance")[gal_inds] #electron abundance of every gas particle in the snapshot (dimensionless)
    neutral_hydrogen_abundance0 = rsf.get_snap_data(snap_name,0,"NeutralHydrogenAbundance")[gal_inds] #neutral hydrogen abundance of every gas particle in the snapshot (dimensionless)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True)[gal_inds] #mass of every gas particle in the snapshot (Msun)
    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0)

    #Find the central index of the subhalo
    neighbour_distances, neighbour_inds = gas_tree.query(subhalopos, k=num_neighbours+1)

    #Find all the edge connections
    # Create dictionary mapping neighbour indices to their position in neighbour_inds
    neighbour_dict = {idx: pos for pos, idx in enumerate(neighbour_inds)}
    # Get list of all indices in pos0
    all_indices = np.arange(len(pos0))
    # Create mapping for non-neighbour indices to 0
    non_neighbour_dict = {idx: 0 for idx in all_indices if idx not in neighbour_inds}
    # Combine both dictionaries
    neighbour_dict.update(non_neighbour_dict)
    smoothing_length = np.max(neighbour_distances)
    edges = find_edges(pos0)
    # Map edge indices to their positions in neighbour_inds using neighbour_dict
    edges = np.array([[neighbour_dict[edge[0]], neighbour_dict[edge[1]]] for edge in edges.T]).T
    # Remove edges that connect to the central particle and are not needed
    edges = edges[:, ~((edges[0] == 0) | (edges[1] == 0))]
    # Subtract 1 from all indices in edges to shift indexing
    edges = edges - 1

    #Neighbour variables (taking out the values for the central particle)
    Ws = spline_kernel(neighbour_distances, smoothing_length)[1:]
    neighbour_pos = pos0[neighbour_inds[1:]] - subhalopos
    neighbour_densities = density0[neighbour_inds[1:]]
    neighbour_velocities = v_gas0[neighbour_inds[1:]] - subhalovel
    neighbour_volumes = volume0[neighbour_inds[1:]]
    neighbour_internal_energy = internal_energy0[neighbour_inds[1:]]
    neighbour_electron_abundance = electron_abundance0[neighbour_inds[1:]]
    neighbour_neutral_hydrogen_abundance = neutral_hydrogen_abundance0[neighbour_inds[1:]]
    neighbour_masses = mass0[neighbour_inds[1:]]

    #calculate temperature
    mu = 4*m_proton/(1+3*X_H+4*X_H*neighbour_electron_abundance)
    neighbour_T = (gamma-1)*neighbour_internal_energy*mu/k_B #temperature of gas cells (K)
    #calculate radial velocity
    neighbour_radial_velocities = np.sum(neighbour_pos*neighbour_velocities, axis=1)/np.sqrt(np.sum(neighbour_pos**2, axis=1))

    #Calculate the mass flux
    mass_flux = np.sum(Ws*neighbour_densities*neighbour_radial_velocities)/np.sum(Ws)

    #Calculate the area of the sphere
    r_inflow = np.sum((3*neighbour_volumes/(4*np.pi))**(1/3)*Ws)/np.sum(Ws)
    area = 4*np.pi*r_inflow**2

    conversion_factor = 3.241028867350146e-17  # 1 km / 1 kpc
    seconds_per_year = 31557600
    mass_flow = area*mass_flux*conversion_factor*seconds_per_year #negative if inflowing. This is in Msun/yr
    return [redshift, subhalomass], mass_flow, np.vstack((neighbour_pos[:,0], neighbour_pos[:,1], neighbour_pos[:,2], neighbour_radial_velocities, neighbour_densities, neighbour_volumes ,neighbour_masses, neighbour_T, neighbour_neutral_hydrogen_abundance)).T, edges

#returns the number of cells within the smoothing length (0.194 ckpc for  high res, 0.387 ckpc for low res)
def num_cells_within_smoothing(snap_dir, snap_number, smoothing_length):
    import numpy as np
    from scipy import spatial
    import read_fof_files as rff
    import read_snap_files as rsf
    import cosmo_utils as cu
    
    #constants
    h = 0.679 #dimensionless Hubble constant

    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)

    #get host group
    host_groups = cu.get_host_groups(snap_dir, snap_number, 0)
    gal_inds = host_groups == 0

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)

    #Need position of the gas particles
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True)[gal_inds] #gas particle positions (kpc)
    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0)

    #Find the central index of the subhalo
    neighbour_inds = gas_tree.query_ball_point(subhalopos, r=smoothing_length)
    return len(neighbour_inds)

