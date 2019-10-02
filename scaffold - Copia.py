
import h5py
#import matplotlib.pyplot as plt
#from scaffold_params import *
#from pyNN.utility.plotting import Figure, Panel
#from ListConnector import *
from spynnaker8 import *
import pyNN.utility.plotting as plot
import matplotlib.pyplot as matplotlib
import numpy as np


def connect_neuron(conn_mat, pre, post, syn_param):

    WEIGHT = syn_param["weight"]
    DELAY = syn_param["delay"]

    #switcher = {'granule': 1117, 'golgi': 2, 'purkinje': 1119, 'basket': 1121, 'stellate': 1123, 'glomerulus': 1150, 'dcn': 1125}
    pre_idx = []
    post_idx = []

    for x in conn_mat[:, 0] + 1:
        #fl = np.where(pre.all_cells - switcher[syn_param["pre"]] == x)
#        fl = np.where(pre.all_cells == x)
        for i in range(0, len(pre.all_cells) -1):
            if (x == i):
                fl = i

        if fl != 0:
            pre_idx.append([fl])

    for x in conn_mat[:, 1] + 1:
        #fl = np.where(post.all_cells - switcher[syn_param["post"]] == x)
        #fl = np.where(post.all_cells == x)

        for i in range(0, len(post.all_cells) -1):
            if (x == i):
                fl = i

        post_idx.append(fl)

    conn_list = []
    for i in range(0, len(pre_idx) -1):
        p = (pre_idx[i][0], post_idx[i], WEIGHT, DELAY)
        conn_list.append(p)

#    conn = ListConnector(conn_list)
    conn = FromListConnector(conn_list)
    ss = StaticSynapse(weight=WEIGHT, delay=DELAY)

    Projection(pre, post, conn, ss)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Set the following variables to 1 to save / plot results
# Set to 0 otherwise
save = 1
plot = 1

########################################################################################
############################### PLACEMENT PARAMETERS ##################################
# The volume base_size can be changed without strict constraints;
# BTW, for single core simulations, size should be kept within 200 - 600 micron range
#
# N.B. by now, for semplicity, keep the two values equal
# volume_base_size = np.array([300., 300.])
base_size = 200.  # 400.
volume_base_size = np.array([base_size, base_size])
dcn_volume = volume_base_size / 2

# Name of data file
filename = 'scaffold_detailed_'
save_name = '{}_{}x{}_v3'.format(filename, volume_base_size[0], volume_base_size[1])
save_name = save_name + '.hdf5'

# Purkinje / DCN ratio: the number of PC per DCN - (Note: describe better)
pc_dcn_ratio = 11.

# Extension of Purkinje cell dendritic tree
pc_extension_dend_tree = 130.
z_pc = 3.5  # NOME DA MODIFICARE - PC extension of the dendritic tree along the z-axis

# Thickness of different layers
layers_thick = {'granular': 150.,
                'purkinje': 30.,
                'molecular': 150.,
                'dcn': 600.}
# Soma radius of each cell type (micron)
''' Diameter of DCN Glutamatergic neuron is in range 15 - 35 micron (Aizemann et al., 2003)
	==> mean diam = 25 micron
	==> mean radius = 12.5 micron
	Slightly different estimate (Gauck and Jaeger, 2000): 10 - 35 micron, average = 20
	==> mean radius = 10 micron'''

cells_radius = {'golgi': 8,
                'glomerulus': 1.5,
                'granule': 2.5,
                'purkinje': 7.5,
                'stellate': 4.,
                'basket': 6.,
                'dcn': 10}

# Density distribution of each cell type
cells_density = {'golgi': 9 * (10 ** (-6)),
                 'glomerulus': 3 * (10 ** (-4)),
                 'granule': 3.9 * (10 ** (-3)),
                 'purkinje': 0.45 * (10 ** (-3)),
                 'stellate': 1.0 / 2 * 10 ** (-4),
                 'basket': 1.0 / 2 * (10 ** (-4))}

# Cell type ID (can be changed without constraints)
cell_type_ID = {'golgi': 1,
                'glomerulus': 2,
                'granule': 3,
                'purkinje': 4,
                'basket': 5,
                'stellate': 6,
                'dcn': 7
                }

# Colors for plots (can be changed without constraints)
cell_color = {'golgi': '#332EBC',
              'glomerulus': '#0E1030',
              'granule': '#E62214',
              'purkinje': '#0F8944',
              'stellate': '#876506',
              'basket': '#7A1607',
              'dcn': '#15118B'}

# Define pc and dcn values once volume base size has been defined
pc_in_volume = int(volume_base_size[0] * volume_base_size[1] * cells_density['purkinje'])
dcn_in_volume = int(pc_in_volume / pc_dcn_ratio)
cells_density['dcn'] = dcn_in_volume / (dcn_volume[0] * dcn_volume[1] * layers_thick['dcn'])

dcn_volume = volume_base_size / 2
save_name = '{}_{}x{}_v3.hdf5'.format(filename, volume_base_size[0], volume_base_size[1])

### Must be generated from previous dictionaries!
# Store positions of cells - organized by cell type
final_cell_positions = {key: [] for key in cell_type_ID.keys()}
placement_stats = {key: {} for key in cell_type_ID.keys()}
for key, subdic in placement_stats.items():
    subdic['number_of_cells'] = []
    subdic['total_n_{}'.format(key)] = 0
    if key != 'purkinje':
        subdic['{}_subl'.format(key)] = 0

########################################################################################
############################### CONNECTOME PARAMETERS ##################################
########################################################################################


# GoC parameters
r_goc_vol = 50  # radius of the GoC volume around the soma
# GoC axon
GoCaxon_z = 30  # max width of GoC axon (keep the minimum possible)
GoCaxon_y = 150  # max height (height of the total simulation volume)
GoCaxon_x = 150  # max lenght

# GrC and parallel fibers parameters
dend_len = 40  # maximum lenght of a GrC dendrite
h_m = 151  # offset for the height of each parallel fiber
sd = 66  # standard deviation of the parallel fibers distribution of heights

# basket and stellate cells parameters
r_sb = 15  # radius of stellate and basket cells area around soma

# Connectivity parameters for granular layer
n_conn_goc = 40  # =n_conn_gloms: number of glomeruli connected to a GoC
n_conn_glom = 4  # number of GoC connected to a glomerulus and number of glomeruli connected to the same GrC
n_connAA = 400  # number of ascending axons connected to a GoC == number of local parallel fibers connected to a GoC
n_conn_pf = 1200  # number of external parallel fibers connected to a GoC
tot_conn = 1600  # total number of conncetions between a GoC and parallel fibers

# Connectivity parameters for granular-molecular layers

# thresholds on inter-soma distances for bc/sc - PCs connectivity
distx = 500
distz = 100
# connectivity parameters
div = 2
conv = 20

# thresholds on inter-soma distances for gap junctions
d_xy = 150
d_z = 50
dc_gj = 4  # divergence = convergence in gap junctions connectivity

# maximum number of connections between parallel fibers and stellate, basket and Purkinje cells
max_conv_sc = 500  # convergence on stellate cells
max_conv_bc = 500  # convergence on basket cells
max_conv_pc = 10000  # convergence on Purkinje cells

# Connectivity parameters for PC-DCN layers
div_pc = 5  # maximum number of connections per PC (typically there are 4-5)

### Data for glom - dcn connectivity
conv_dcn = 147  # convergence
div_dcn = 2  # or 3 - to be tested - divergence


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



DELAY = 1
setup(timestep= DELAY, min_delay= DELAY)

filename = 'scaffold_detailed__158.0x158.0_v3.hdf5'
#filename = 'scaffold_full_dcn_400.0x400.0_v3.hdf5'
f = h5py.File(filename, 'r+')
positions = np.array(f['positions'])
num_tot = 0

sorted_nrn_types = sorted(list(cell_type_ID.values()))

# Create a 'inverse map' of cell_type_IDaa_pc
id_2_cell_type = {val: key for key, val in cell_type_ID.items()}

# Create a dictionary with all cell names (keys)
# and lists that will contain nest models (values)
neuron_models = {key: [] for key in cell_type_ID.keys()}

for cell_id in sorted_nrn_types:
    cell_name = id_2_cell_type[cell_id]
    if cell_name != 'glomerulus':

        if cell_name == 'golgi':
            cell_params = {'tau_refrac': 2.0,  # ms
                                         'cm': 0.076,  # nF
                                         'v_thresh': -55.0,  # mV
                                         'v_reset': -75.0,  # mV
                                         'tau_m': 21.1,
                                         #'e_rev_leak': -65.0,  # mV
                                         'i_offset': 36.75,  # pA # tonic ~9-10 Hz  ;previous = 36.0 pA
                                         'tau_syn_E': 0.5,
                                         'tau_syn_I': 10.0}
        elif cell_name == 'granule':
            cell_params = {'tau_refrac': 1.5,  # ms
                                         'cm': 0.003,  # nF
                                         'v_thresh': -42.0,  # mV
                                         'v_reset': -84.0,  # mV
                                         'tau_m': 20,
                                         #'v_rest': -74.0,  # mV
                                         'i_offset': 0,  # pA # tonic ~9-10 Hz  ;previous = 36.0 pA
                                         'tau_syn_E': 0.5,
                                         'tau_syn_I': 10.0}

            cell_pos = positions[positions[:, 1] == cell_id, :]
            pop_gr = Population(cell_pos.shape[0], model)

        elif cell_name == 'purkinje':
            cell_params = {'tau_refrac': 0.8,  # ms
                                         'cm': 0.062,  # nF
                                         'v_thresh': -47.0,  # mV
                                         'v_reset': -72.0,  # mV
                                         'tau_m': 88.6,
                                         #'e_rev_leak': -62.0,  # mV
                                         'i_offset': 750,  # pA # tonic ~9-10 Hz  ;previous = 36.0 pA
                                         'tau_syn_E': 0.5,
                                         'tau_syn_I': 10.0}
        elif cell_name == 'stellate' or cell_name == 'basket':
            cell_params = {'tau_refrac': 1.59,  # ms
                                         'cm': 0.0146,  # nF
                                         'v_thresh': -53.0,  # mV
                                         'v_reset': -78.0,  # mV
                                         'tau_m': 14.6,
                                         #'e_rev_leak': -68.0,  # mV
                                         'i_offset': 15.6,  # pA # tonic ~9-10 Hz  ;previous = 36.0 pA
                                         'tau_syn_E': 0.5,
                                         'tau_syn_I': 10.0}
        elif cell_name == 'dcn':
            cell_params = {'tau_refrac': 3.7,  # ms
                                         'cm': 0.089,  # nF
                                         'v_thresh': -48.0,  # mV
                                         'v_reset': -69.0,  # mV
                                         'tau_m': 57.1,
                                         #'e_rev_leak': -59.0,  # mV
                                         'i_offset': 45.75,  # pA # tonic ~9-10 Hz  ;previous = 36.0 pA
                                         'tau_syn_E': 0.5,
                                         'tau_syn_I': 10.0}

        model = IF_cond_exp(**cell_params)

    else:
        #model = SpikeSourcePoisson()
        model = IF_cond_exp(**cell_params)

    cell_pos = positions[positions[:, 1] == cell_id, :]
    neuron_models[cell_name] = Population(cell_pos.shape[0], model)

    num_tot += cell_pos.shape[0]


#noise = NoisyCurrentSource(mean=0, stdev=50)
#print(neuron_models.get('dcn'))
#print(neuron_models['dcn'].all_cells)
#print(neuron_models['dcn'][1])


#for name in neuron_models.keys():
#    a = neuron_models.get(name)
#    a.inject(noise)
#    noise.inject_into(cell)


conn_aa_goc = np.array(f['connections/aa_goc'])
syn_p = {"model": "static_synapse", "weight": 20, "delay": 0.5, "pre": "granule", "post": "golgi"}
connect_neuron(conn_aa_goc, neuron_models.get('granule'), neuron_models.get('golgi'), syn_p)

conn_aa_pc = np.array(f['connections/aa_pc'])
syn_p = {"model" : "static_synapse", "weight" : 50.0, "delay": 0.5, "pre": "granule", "post": "purkinje"}
#connect_neuron(conn_aa_pc, neuron_models.get('granule'), neuron_models.get('purkinje'), syn_p)

conn_bc_pc = np.array(f['connections/bc_pc'])
syn_p = {"model" : "static_synapse", "weight" : -2.50, "delay": 0.5, "pre": "basket", "post": "purkinje"}
#connect_neuron(conn_bc_pc, neuron_models.get('basket'), neuron_models.get('purkinje'), syn_p)

conn_gj_bc = np.array(f['connections/gj_bc'])
syn_p = {"model" : "static_synapse", "weight" : -3, "delay": 0.5, "pre": "basket", "post": "basket"}
#connect_neuron(conn_gj_bc, neuron_models.get('basket'), neuron_models.get('basket'), syn_p)

conn_gj_sc = np.array(f['connections/gj_sc'])
syn_p = {"model": "static_synapse", "weight": -2.5, "delay": 0.5, "pre": "stellate", "post": "stellate"}
#connect_neuron(conn_gj_sc, neuron_models.get('stellate'), neuron_models.get('stellate'), syn_p)

conn_glom_goc = np.array(f['connections/glom_goc'])
syn_p = {"model": "static_synapse", "weight": 2.0, "delay": 0.5, "pre": "glomerulus", "post": "golgi"}
#connect_neuron(conn_glom_goc, neuron_models.get('glomerulus'), neuron_models.get('golgi'), syn_p)
#syn_p = {"model": "static_synapse", "weight": 2.0, "delay": 0.5, "pre": "0", "post": "golgi"}
#connect_neuron(conn_glom_goc, stimulus, neuron_models.get('golgi'), syn_p)

conn_glom_grc = np.array(f['connections/glom_grc'])
syn_p = {"model": "static_synapse", "weight": 9.0, "delay": 0.5, "pre": "glomerulus", "post": "granule"}
#connect_neuron(conn_glom_grc, neuron_models.get('glomerulus'), neuron_models.get('granule'), syn_p)

conn_goc_grc = np.array(f['connections/goc_grc'])
syn_p = {"model": "static_synapse", "weight": -5, "delay": 0.5, "pre": "golgi", "post": "granule"}
#connect_neuron(conn_goc_grc, neuron_models.get('golgi'), neuron_models.get('granule'), syn_p)

conn_pc_dcn = np.array(f['connections/pc_dcn'])
syn_p = {"model": "static_synapse", "weight": -0.2, "delay": 0.5, "pre": "purkinje", "post": "dcn"}
#connect_neuron(conn_pc_dcn, neuron_models.get('purkinje'), neuron_models.get('dcn'), syn_p)

conn_pf_bc = np.array(f['connections/pf_bc'])
syn_p = {"model": "static_synapse", "weight": 0.4, "delay": 1, "pre": "granule", "post": "basket"}
#connect_neuron(conn_pf_bc, neuron_models.get('granule'), neuron_models.get('basket'), syn_p)

conn_pf_goc = np.array(f['connections/pf_goc'])
syn_p = {"model": "static_synapse", "weight": 0.4, "delay": 1, "pre": "granule", "post": "golgi"}
#connect_neuron(conn_pf_goc, neuron_models.get('granule'), neuron_models.get('golgi'), syn_p)

conn_pf_pc = np.array(f['connections/pf_pc'])
syn_p = {"model": "static_synapse", "weight": 0.05, "delay": 1, "pre": "granule", "post": "purkinje"}
#connect_neuron(conn_pf_pc, neuron_models.get('granule'), neuron_models.get('purkinje'), syn_p)

conn_pf_sc = np.array(f['connections/pf_sc'])
syn_p = {"model": "static_synapse", "weight": 0.4, "delay": 1, "pre": "granule", "post": "stellate"}
#connect_neuron(conn_pf_sc, neuron_models.get('granule'), neuron_models.get('stellate'), syn_p)

conn_sc_pc = np.array(f['connections/sc_pc'])
syn_p = {"model": "static_synapse", "weight": -2, "delay": 0.5, "pre": "stellate", "post": "purkinje"}
#connect_neuron(conn_sc_pc, neuron_models.get('stellate'), neuron_models.get('purkinje'), syn_p)

conn_glom_dcn = np.array(f['connections/glom_dcn'])
syn_p = {"model": "static_synapse", "weight": 2.0, "delay": 0.5, "pre": "glomerulus", "post": "dcn"}
#connect_neuron(conn_glom_dcn, neuron_models.get('glomerulus'), neuron_models.get('dcn'), syn_p)

conn_gj_goc = np.array(f['connections/gj_goc'])
syn_p = {"model": "static_synapse", "weight": -8.0, "delay": 0.5, "pre": "golgi", "post": "golgi"}
#connect_neuron(conn_gj_goc, neuron_models.get('golgi'), neuron_models.get('golgi'), syn_p)

###########  Stim and simulation

TOT_DURATION = 500.  # mseconds
STIM_START = 300.  # beginning of stimulation
STIM_END = 350.  # end of stimulation
STIM_FREQ = 120.  # Frequency in Hz
RADIUS = 50.5  # Microm

#setup(timestep= 1.0)
spike_nums = np.int(np.round((STIM_FREQ * (STIM_END - STIM_START)) / TOT_DURATION))
#spike_nums = len(neuron_models.get('golgi'))
stim_array = np.round(np.linspace(STIM_START, STIM_END, spike_nums))
stim_array_int = []
for i in stim_array:
    stim_array_int.append(int(i))
origin = np.array([200., 200., 75.])
phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
x = RADIUS * np.sin(phi) * np.cos(theta) + origin[0]
y = RADIUS * np.sin(phi) * np.sin(theta) + origin[1]
z = RADIUS * np.cos(phi) + origin[2]

gloms_pos = positions[positions[:, 1] == cell_type_ID['glomerulus'], :]
# find center of 'glomerular sphere'
x_c, y_c, z_c = np.median(gloms_pos[:, 2]), np.median(gloms_pos[:, 3]), np.median(gloms_pos[:, 4])

# Find glomeruli falling into the selected volume
target_gloms_idx = np.sum((gloms_pos[:, 2::] - np.array([x_c, y_c, z_c])) ** 2, axis=1).__lt__(RADIUS ** 2)
tg_idx = []
#for i in range(neuron_models.get('granule').size): #range(len(target_gloms_idx)):
#    if target_gloms_idx[i] == True:
#        tg_idx.append(i)

#target_gloms = gloms_pos[tg_idx, 0] + 1
#print(neuron_models['glomerulus'].all_cells)
#id_stim = [glom for glom in neuron_models['glomerulus'].all_cells if glom in target_gloms]

#count = 0
#id_stim = []
#for glom in neuron_models['glomerulus'].all_cells:
#    print(glom in target_gloms)
#    if(glom in target_gloms):
#        id_stim.append(count)
#    for i in target_gloms:
#        if(i == glom):
#            id_stim.append(count)
#    count += 1


#n = len(target_gloms)

print(neuron_models.get('granule').size)
#spikeTimes= [2, 4, 6, 8, 10]
stimulus = Population(neuron_models.get('granule').size, SpikeSourceArray(spike_times=stim_array_int))
#neuron_models.get('granule').size
conn_list = []

#for i in range(n):
#    p = (0, tg_idx[i], 1., 1)
#    conn_list.append(p)

#conn = ListConnector(conn_list)
ss = StaticSynapse(weight=1)
#Projection(stimulus, neuron_models.get('glomerulus'), conn, ss)
#Projection(stimulus, neuron_models.get('granule'), conn, ss)
#Projection(stimulus, neuron_models.get('glomerulus')[tg_idx], AllToAllConnector(allow_self_connections=False), ss)

#stimulus = Population(1, SpikeSourceArray(spike_times= stim_array_int))
Projection(stimulus, neuron_models.get('granule'), OneToOneConnector(), ss)

#for i in neuron_models.values():
#    i.record(['v'])
neuron_models.get('golgi').record(['spikes', 'v'])

run(TOT_DURATION)


pop_1 = neuron_models.get('golgi')
#data1 = pop_gr.get_data(variables=["v"])

neo = pop_1.get_data(variables=["spikes", "v"])
spikes = neo.segments[0].spiketrains
v = neo.segments[0].filter(name='v')[0]

#Figure(Panel(g_data.get_data('spikes')), title="Prova")
#print(data1)
simtime = 10
plot.Figure(
        # plot voltage for first ([0]) neuron
        plot.Panel(v[0], ylabel="Membrane potential (mV)",
                   data_labels=[pop_1.label], yticks=True, xlim=(0, simtime)),
                    # plot spikes (or in this case spike)
                    #plot.Panel(spikes, yticks=True, xlim=(0, simtime)),
                    #title="Simple Example",
                    #annotations="Simulated with {}".format(name())
        )
matplotlib.show()