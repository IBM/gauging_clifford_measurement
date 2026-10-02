# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import stim
import matplotlib.pyplot as plt
from matplotlib import patches

def qubit_coord_groups(d):
    qubit_dict =  {}
    ancilla_dict = {}
    q_ind = 0
    a_ind = 0
    even_row = 0
    for col in range(1,3*d):
        if 3*d-4>col>=-2.5:
            if (col%6 in (1,3)):
                qubit_dict[(-1, col)] = q_ind
                q_ind += 1
            if (col%6 in (0,2,4)):
                ancilla_dict[(-1, col)] = a_ind
                a_ind += 1

    for row in range(int(3*(d-1)/2)+1):
        even_row = row % 2 == 0
        for col in range(1,3*d-1):
            if 3*d-row-1+(row%3==2)-(row%3==0)>col>=row-0.5-1-(row%3==0)+(row%3==1):
                if (col%6 in (0,4) and even_row) or (col%6 in (1,3) and not even_row):
                    qubit_dict[(row, col)] = q_ind
                    q_ind += 1
                if (even_row and col%6 in (1,2,3,5)) or (not even_row and col%6 in (0,2,4,5)):
                    ancilla_dict[(row, col)] = a_ind
                    a_ind += 1
    ancilla_dict = {k:v+len(qubit_dict) for k,v in ancilla_dict.items()}

    flag_qubit_coords = [coord for coord in ancilla_dict.keys() if (coord[0]%2==0 and coord[1]%6 == 2) or (coord[0]%2==1 and coord[1]%6 == 5)]
    assert len(flag_qubit_coords)%3==0 #same number of plaquettes from each color

    link_coords = []
    for coord in ancilla_dict.keys():
        if ((coord[0]%2==0 and coord[1]%6 == 3)
            or (coord[0]%2==1 and coord[1]%6 == 0)):
            #above the flag
            q_coord_right = coord[0]+1,coord[1] #always exists
            q_coord_above = coord[0],coord[1]+1
            q_coord_left = coord[0]-1,coord[1]
            flag_coord = (coord[0],coord[1]-1)
            if q_coord_above in qubit_dict:
                link_coords.append(((q_coord_above,q_coord_right),coord,flag_coord))
            else:
                link_coords.append(((q_coord_right, q_coord_left),coord,flag_coord))

        if ((coord[0]%2==0 and coord[1]%6 == 1)
            or (coord[0]%2==1 and coord[1]%6 == 4)):
            #below the flag
            q_coord_right = coord[0]+1,coord[1] #always exists
            q_coord_below = coord[0],coord[1]-1
            q_coord_left = coord[0]-1,coord[1]
            flag_coord = (coord[0],coord[1]+1)
            if q_coord_below in qubit_dict:
                link_coords.append(((q_coord_right, q_coord_below),coord,flag_coord))
            else:
                link_coords.append(((q_coord_right, q_coord_left),coord,flag_coord))

        if (coord[0]%2==1 and coord[1]%6 == 2) or (coord[0]%2==0 and coord[1]%6 == 5):
            #in-between two flags
            flag_coord = (coord[0]+1,coord[1])
            link_coords.append((((coord[0],coord[1]-1),(coord[0],coord[1]+1)),coord,flag_coord))

    assert len(link_coords) == len(ancilla_dict)-len(flag_qubit_coords) #every non-flag ancilla is included in one link


    boundary_qubit_coords = []
    for coord in qubit_dict.keys():
        if ((coord[0]<0 and coord!=(-1,1))
            or (coord[1] > 3*d-2-coord[0])
            or (coord[1] < coord[0]-1)):
            boundary_qubit_coords+=[coord]


    plaquettes = []
    for center in flag_qubit_coords:
        plaquette = {}
        plaquette['flag_coord'] = center
        plaquette['ancilla1_coord'] = center[0],center[1]-1
        plaquette['qubits1_coord'] = [(center[0]+dx, center[1]+dy) 
                                      if (center[0]+dx, center[1]+dy) in qubit_dict 
                                            and (center[0]+dx, center[1]+dy) not in boundary_qubit_coords
                                            else None
                                    for dx,dy in [(+1,-1), (0,-2), (-1,-1)]
                                        
                                    ]

        plaquette['ancilla2_coord'] = center[0],center[1]+1
        plaquette['qubits2_coord'] = [(center[0]+dx, center[1]+dy)
                                        if (center[0]+dx, center[1]+dy) in qubit_dict 
                                            and (center[0]+dx, center[1]+dy) not in boundary_qubit_coords
                                            else None
                                    for dx,dy in [(+1,+1), (0,+2), (-1,+1)]
                                    ]
        plaquettes.append(plaquette)
    
    return qubit_dict, ancilla_dict, flag_qubit_coords, link_coords, boundary_qubit_coords, plaquettes

def draw_lattice(d,qubit_coords,ancilla_coords, plaquettes = None, qubit_label = True, no_show = False, frame_on = True):
    fig, ax = plt.subplots()

    ax.set_frame_on(frame_on)
    ax.get_xaxis().set_visible(frame_on)
    ax.get_yaxis().set_visible(frame_on)

    ax.plot(np.array(list(qubit_coords.keys())).T[1],-np.array(list(qubit_coords.keys())).T[0], "o", mfc="gray", mec="gray")
    if qubit_label:
        for coord, label in qubit_coords.items():
            ax.text(coord[1],-coord[0], str(label))

    # drawing boundaries
    xmax = int(3*(d-1)/2)+1
    ax.plot((1, xmax-1),(-1, -xmax+1), "green")
    ax.plot((1, 1), (-1, 1), "green")
    ax.plot((3*d-3,3*d-2-xmax), (0, -xmax+1), "red")
    ax.plot((4, 3*d-3), (0, 0), "blue")
    ax.plot((4, 1), (0, 1), "blue")
    fig.set_size_inches(12, 7)

    ax.plot(np.array(list(ancilla_coords.keys())).T[1],-np.array(list(ancilla_coords.keys())).T[0], "o", mfc="none", mec="gray")
    if qubit_label:
        for coord, label in ancilla_coords.items():
            ax.text(coord[1],-coord[0], str(label))
        
    plaquettes=[{k:(v[1],-v[0]) if len(v)==2 else [(coord[1],-coord[0]) if coord else None for coord in v] for k,v in plaq.items()} for plaq in plaquettes]

    if plaquettes:
        for plaquette in plaquettes:
            qubit_coords = [coord for coord in plaquette['qubits1_coord'] if coord] + [coord for coord in plaquette['qubits2_coord'] if coord][::-1]

            facecolor = "blue"
            center = plaquette['flag_coord']
            if center[1]%6 == 0 and center[0]%6==2:
                facecolor = "red"
            if center[1]%6 == 3 and center[0]%6==5:
                facecolor = "red"

            if center[1]%6 == 1 and center[0]%6==5:
                facecolor = "green"
            if center[1]%6 == 4 and center[0]%6==2:
                facecolor = "green"

            polygon = patches.Polygon(qubit_coords,
                                        closed=True,
                                        facecolor=facecolor,
                                        linewidth=0,
                                        alpha=0.25,
                                    )
            ax.add_patch(polygon)
    
    if not no_show:
        plt.show()

def add_noise(stim_circuit: stim.Circuit, CX_error=0, single_qubit_error=0, meas_error=0, gate_idle_error=0, meas_idle_error=0):
    if not meas_idle_error:
        meas_idle_error = gate_idle_error
    noisy_circuit = stim.Circuit()
    i=0
    while i<len(stim_circuit):
        idle_qubits = set(list(range(stim_circuit.num_qubits)))
        meas_round = False
        disable_noise = False
        mpp_round = False
        while i<len(stim_circuit) and stim_circuit[i].name!="TICK":
            inst: stim.CircuitInstruction = stim_circuit[i]
            if inst.tag=='disable_noise' and not disable_noise:
                disable_noise = True
            if disable_noise:
                    CX_error, single_qubit_error, meas_error, gate_idle_error, meas_idle_error = (0,0,0,0,0)
            trgs = inst.targets_copy()
            qubit_trgs = [trg.qubit_value for trg in trgs if trg.is_qubit_target]
            if inst.name == "QUBIT_COORDS":
                noisy_circuit.append(inst)
            elif inst.name == "CX":
                noisy_circuit.append(inst)
                if trgs[0].is_qubit_target:
                    if CX_error:
                        noisy_circuit.append("DEPOLARIZE2",qubit_trgs,CX_error)

            elif inst.name in ["S","S_DAG"]:
                noisy_circuit.append(inst)
                if single_qubit_error:
                    noisy_circuit.append("DEPOLARIZE1",qubit_trgs,single_qubit_error)
            elif inst.name in ["M","MX","R","RX"]:
                meas_round = True
                if inst.name in ["M","MX"]:
                    noisy_circuit.append(inst.name,qubit_trgs,meas_error)
                    noisy_circuit.append("DEPOLARIZE1",qubit_trgs,meas_error)
                else:
                    noisy_circuit.append(inst)
                    if meas_error:
                        if inst.name=="R":
                            noisy_circuit.append("X_ERROR",qubit_trgs,meas_error)
                        else:
                            noisy_circuit.append("Z_ERROR",qubit_trgs,meas_error)
            else:
                noisy_circuit.append(inst)
            if inst.name == "MPP":
                mpp_round = True
            idle_qubits-=set(qubit_trgs)
            i+=1
        if i<len(stim_circuit)-1 and stim_circuit[i].name=="TICK" and stim_circuit[i+1].name=="TICK":
                # double TICK marks the start of the final noiseless rounds
                CX_error, single_qubit_error, meas_error, gate_idle_error, meas_idle_error = (0,0,0,0,0)
        else:
            if meas_round:
                if meas_idle_error and not mpp_round:
                        noisy_circuit.append("DEPOLARIZE1",idle_qubits,meas_idle_error)
            elif gate_idle_error:
                if not mpp_round:
                    noisy_circuit.append("DEPOLARIZE1",idle_qubits,gate_idle_error)
        
        noisy_circuit.append("TICK")
        i+=1
    return noisy_circuit
