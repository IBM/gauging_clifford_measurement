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

def qubit_coord_groups_escape(d, d_sc = 11):
    qubit_dict =  {}
    ancilla_dict = {}
    q_ind = 0
    a_ind = 0
    even_row = 0

    for row in range(int(3*(d-1)/2)+1):
        even_row = row % 2 == 0
        for col in range(0,3*d-1):
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

    grafted_Xplaquettes = []
    grafted_Zplaquettes = []

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
        grafted_Xplaquettes.append({"ancilla_coord":center,"qubits_coord":plaquette['qubits1_coord'] +plaquette['qubits2_coord'][::-1]})
        grafted_Zplaquettes.append({"ancilla_coord":center,"qubits_coord":plaquette['qubits1_coord'] +plaquette['qubits2_coord'][::-1]})
    

    sc_corner = np.array([0,3*d-3])
    y_step = np.array([1,-1])
    x_step = np.array([-1,-1])

    sc_ancilla_coords = []
    q_ind+=a_ind
    sc_logical_X = []
    sc_logical_Z = []
    for y in range(d_sc):
        for x in range(d_sc+(d_sc%2==0)*(y%2==0)):
            coord = sc_corner+x*x_step+y*y_step
            if tuple(coord) not in qubit_dict and tuple(coord) not in ancilla_dict and x<d_sc:
                qubit_dict[tuple(coord)] = q_ind
                q_ind+=1
            if tuple(coord) in qubit_dict and y==d_sc-1:
                sc_logical_Z+=[qubit_dict[tuple(coord)]]
            if tuple(coord) in qubit_dict and x==d_sc-1:
                sc_logical_X+=[qubit_dict[tuple(coord)]]

    a_ind=q_ind
    for y in range(d_sc):
        for x in range(d_sc+(d_sc%2==0)*(y%2==0)):
            coord = sc_corner+x*x_step+y*y_step
            coord += np.array([(x%2)-1,-(x%2)])
            coord += -(1-(y+x%2)%2)*x_step + ((y+x%2)==0)*x_step
            if coord[0] > -d_sc and not (x==d_sc-1+(d_sc%2) and y%2==0):
                sc_ancilla_coords.append(tuple(coord))
                if tuple(coord) not in ancilla_dict:
                    ancilla_dict[tuple(coord)] = a_ind
                    a_ind+=1

    sc_plaquettes = []
    qubit_rel_coords = [np.array([1,0]),np.array([0,1]),np.array([-1,0]),np.array([0,-1])]
    for a_coord in sc_ancilla_coords:
        plaquette = {}
        plaquette["ancilla_coord"] = a_coord
        plaquette["qubits_coord"] = [tuple(np.array(a_coord)+rel_coord) 
                                     for rel_coord in qubit_rel_coords 
                                     if tuple(np.array(a_coord)+rel_coord) in qubit_dict and not (tuple(np.array(a_coord)+rel_coord) in boundary_qubit_coords and a_coord[1]>3*d/2)
                                     or tuple(np.array(a_coord)+rel_coord) in flag_qubit_coords]

        if plaquette["qubits_coord"]:
            sc_plaquettes.append(plaquette)
            xmax = int(3*(d-1)/2)
            if set(plaquette["qubits_coord"]).intersection(set(flag_qubit_coords)) == set() and not ((a_coord[0]%6,a_coord[1]%6)==(1,0) and a_coord[0]//6 == a_coord[1]//6>=0):
                if set(plaquette["qubits_coord"]).intersection(set([(x,3*d-3-x) for x in range(1,xmax)])) == set():
                    if a_coord[0]%2:
                        grafted_Xplaquettes.append(plaquette)
                    else:
                        grafted_Zplaquettes.append(plaquette)

    # parallel to X boundary
    X1_stab_anc_coord = [(coord[0]-1,coord[1])
                        for coord in boundary_qubit_coords if coord[1]<3*(d+1)/2 and coord[0]%2==0]
    X2_stab_anc_coord = [(coord[0],coord[1]+1)
                        for coord in boundary_qubit_coords if coord[1]<3*(d+1)/2 and not coord[0]%2==0] 
    for plaq in grafted_Xplaquettes:
        a1_coord = tuple(np.array(plaq["ancilla_coord"]) +(0,-1))
        a2_coord = tuple(np.array(plaq["ancilla_coord"]) +(1,0))
        if a1_coord in X1_stab_anc_coord:
            q_coords = [tuple(np.array(plaq["ancilla_coord"]) +(1,-1)),tuple(np.array(plaq["ancilla_coord"]) +(0,-2))]
            plaq["qubits_coord"] = plaq["qubits_coord"] + q_coords
        if a2_coord in X2_stab_anc_coord:
            q_coords = [tuple(np.array(plaq["ancilla_coord"]) +(2,0)),tuple(np.array(plaq["ancilla_coord"]) +(1,-1))]
            plaq["qubits_coord"] = plaq["qubits_coord"] + q_coords
        if plaq["ancilla_coord"] == (0,2):
            plaq["qubits_coord"] = [(1, 1), (2, 0), (1, -1), (0, 0), (0, 4), (1, 3)]
        if plaq["ancilla_coord"][0]>=0 and (plaq["ancilla_coord"][0]%6,plaq["ancilla_coord"][1]%6)== (0,2) and (plaq["ancilla_coord"][0]//6==plaq["ancilla_coord"][1]//6!=0):
            plaq["qubits_coord"] = [(plaq["ancilla_coord"][0]+1, plaq["ancilla_coord"][1]-1),
                                    (plaq["ancilla_coord"][0]+2, plaq["ancilla_coord"][1]-2),
                                    (plaq["ancilla_coord"][0]+1, plaq["ancilla_coord"][1]-3),
                                    (plaq["ancilla_coord"][0], plaq["ancilla_coord"][1]-2),
                                    (plaq["ancilla_coord"][0]-1, plaq["ancilla_coord"][1]-1),
                                    (plaq["ancilla_coord"][0]-1, plaq["ancilla_coord"][1]+1),
                                    (plaq["ancilla_coord"][0], plaq["ancilla_coord"][1]+2),
                                    (plaq["ancilla_coord"][0]+1, plaq["ancilla_coord"][1]+1)]


    # diagonal edge
    Z_stab_anc_coord =  [coord for coord in ancilla_dict if 0<coord[1] and coord[0]==0]
    for plaq in grafted_Zplaquettes:
        a1_coord = tuple(np.array(plaq["ancilla_coord"]) +(-1,0))
        if plaq["ancilla_coord"] in  Z_stab_anc_coord:
            q_coords = [tuple(np.array(plaq["ancilla_coord"]) +rc) for rc in [(+1,-1), (0,-2), (-1,-1), (-1,1), (0,2),(1,1)]]
            plaq["qubits_coord"] = q_coords
        if a1_coord in  Z_stab_anc_coord:
            q_coords = [tuple(np.array(plaq["ancilla_coord"]) +(-2,0))]
            plaq["qubits_coord"] = plaq["qubits_coord"][:3] + q_coords +  plaq["qubits_coord"][3:]

    # triangles excluding a flag
    triangle_Zancilla_coords = [(coord[0]-1,coord[1]) if coord[0]%2 else (coord[0],coord[1]+1)
                               for coord in boundary_qubit_coords if coord[1]<3*(d+1)/2]
    triangle_Xancilla_coords = [(coord[0]-1,coord[1]) for coord in flag_qubit_coords if 0<coord[1] and coord[0]==0]
    for a_coord in triangle_Zancilla_coords:
        qubit_coords = [tuple(np.array(a_coord)+rel_coord) 
                                     for rel_coord in qubit_rel_coords 
                                     if tuple(np.array(a_coord)+rel_coord) not in flag_qubit_coords]
        grafted_Zplaquettes.append({"ancilla_coord":a_coord,"qubits_coord":qubit_coords})

    for a_coord in triangle_Xancilla_coords:
        qubit_coords = [tuple(np.array(a_coord)+rel_coord) 
                                     for rel_coord in qubit_rel_coords 
                                     if tuple(np.array(a_coord)+rel_coord) not in flag_qubit_coords]
        grafted_Xplaquettes.append({"ancilla_coord":a_coord,"qubits_coord":qubit_coords})
    
    return qubit_dict, ancilla_dict, flag_qubit_coords, link_coords, boundary_qubit_coords, plaquettes, sc_plaquettes, sc_logical_X, sc_logical_Z, grafted_Xplaquettes, grafted_Zplaquettes

def draw_lattice_escape(d,qubit_coords,ancilla_coords, plaquettes = None, X_plaquettes = None, Z_plaquettes = None, qubit_label = True, no_show = False, frame_on = True):
    fig, ax = plt.subplots(figsize=(10,10))

    ax.set_frame_on(frame_on)
    ax.get_xaxis().set_visible(frame_on)
    ax.get_yaxis().set_visible(frame_on)

    ax.plot(np.array(list(qubit_coords.keys())).T[1],-np.array(list(qubit_coords.keys())).T[0], "o", mfc="gray", mec="gray")
    if qubit_label:
        for coord, label in qubit_coords.items():
            ax.text(coord[1],-coord[0], str(label))

    ax.plot(np.array(list(ancilla_coords.keys())).T[1],-np.array(list(ancilla_coords.keys())).T[0], "o", mfc="none", mec="gray")
    if qubit_label:
        for coord, label in ancilla_coords.items():
            ax.text(coord[1],-coord[0], str(label))
        

    if X_plaquettes:
        for plaquette in X_plaquettes:
            qubit_coords = [(coord[1],-coord[0]) for coord in plaquette['qubits_coord'] if coord]
            if len(qubit_coords)==2:
                qubit_coords.append((plaquette['ancilla_coord'][1],-plaquette['ancilla_coord'][0]))
            polygon = patches.Polygon(qubit_coords,
                                        closed=True,
                                        facecolor="lightgray",
                                        edgecolor='black',
                                        linewidth=0.5,
                                        alpha=0.4,
                                    )
            ax.add_patch(polygon)
    if Z_plaquettes:
        for plaquette in Z_plaquettes:
            qubit_coords = [(coord[1],-coord[0]) for coord in plaquette['qubits_coord'] if coord]
            if len(qubit_coords)==2:
                qubit_coords.append((plaquette['ancilla_coord'][1],-plaquette['ancilla_coord'][0]))
            polygon = patches.Polygon(qubit_coords,
                                        closed=True,
                                        facecolor="black",
                                        edgecolor='black',
                                        linewidth=0.5,
                                        alpha=0.2,
                                    )
            ax.add_patch(polygon)

    if plaquettes:
        plaquettes=[{k:(v[1],-v[0]) if len(v)==2 else [(coord[1],-coord[0]) if coord else None for coord in v] for k,v in plaq.items()} for plaq in plaquettes]
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
                                        edgecolor='black',
                                        linewidth=0.5,
                                        alpha=.25,
                                    )
            ax.add_patch(polygon)



    if not no_show:
        plt.show()

class EscapeColorCode:
    def __init__(self, d_cc, d_sc):
        self.d_cc = d_cc
        self.d_sc = d_sc
        self.no_hadamards = True
        self.pipeline = False
        self.grow = False

        self.qubit_dict, self.ancilla_dict, self.flag_qubit_coords, _, _, self.cc_plaquettes, self.sc_plaquettes, self.sc_logical_X, self.sc_logical_Z, self.grafted_Xplaquettes, self.grafted_Zplaquettes = qubit_coord_groups_escape(d_cc, d_sc)
        self.every_qubit_set = set(list(self.qubit_dict.values())+list(self.ancilla_dict.values()))
            
        self.circuit = stim.Circuit()
        for coord, q in self.qubit_dict.items():
            self.circuit.append("QUBIT_COORDS",q,[coord[1]+2*d_sc-3*(d_cc-1)-2,coord[0]+d_sc-1])
            # self.circuit.append("QUBIT_COORDS",q,[coord[0]+d_sc,coord[1]+d_sc-d_cc])
        for coord, q in self.ancilla_dict.items():
            self.circuit.append("QUBIT_COORDS",q,[coord[1]+2*d_sc-3*(d_cc-1)-2,coord[0]+d_sc-1])
            # self.circuit.append("QUBIT_COORDS",q,[coord[0]+d_sc,coord[1]+d_sc-d_cc])

        self.stim_polygon_comments = ""
        for plaquette in self.cc_plaquettes:
            qubits = [self.qubit_dict[coord] for coord in plaquette['qubits1_coord'] if coord] + [self.qubit_dict[coord] for coord in plaquette['qubits2_coord'] if coord][::-1]

            facecolor = "#!pragma POLYGON(1,1,0,0.25)"
            center = plaquette['flag_coord']
            if center[0]%6 == 0 and center[1]%6==2:
                facecolor = "#!pragma POLYGON(1,0,1,0.25)"
            if center[0]%6 == 3 and center[1]%6==5:
                facecolor = "#!pragma POLYGON(1,0,1,0.25)"

            if center[0]%6 == 1 and center[1]%6==5:
                facecolor = "#!pragma POLYGON(0,1,1,0.25)"
            if center[0]%6 == 4 and center[1]%6==2:
                facecolor = "#!pragma POLYGON(0,1,1,0.25)"

            stim_polygon=facecolor
            for q in qubits:
                stim_polygon+=" "+str(q)

            self.stim_polygon_comments+=stim_polygon+"\n"
        

        self.measurement_dict = {}
        self.meas_ind = 0
        self.time = 0
        self.last_logmeas_time = None
        
        self.measure_Clifford_on_data = False

    def grow_from_cc(self, basis = "Z", measure_sc_plaquettes=True):
        self.grow = True
        for stab in self.cc_plaquettes:
            q_inds = [self.qubit_dict[c] if c in self.qubit_dict else self.ancilla_dict[c] for c in stab['qubits1_coord']+stab['qubits2_coord'] if c]
            a_ind = self.ancilla_dict[stab['flag_coord']]
            Pstring = 'X'+str(q_inds[0])
            for q in q_inds[1:]:
                Pstring += '*X'+str(q)
            self.circuit.append("MPP",stim.PauliString(Pstring))
            self.measurement_dict[("X",a_ind,self.time)] = self.meas_ind
            self.meas_ind += 1
        self.circuit.append("TICK")
        for stab in self.cc_plaquettes:
            q_inds = [self.qubit_dict[c] if c in self.qubit_dict else self.ancilla_dict[c] for c in stab['qubits1_coord']+stab['qubits2_coord'] if c]
            a_ind = self.ancilla_dict[stab['flag_coord']]
            Pstring = 'Z'+str(q_inds[0])
            for q in q_inds[1:]:
                Pstring += '*Z'+str(q)
            self.circuit.append("MPP",stim.PauliString(Pstring))
            self.measurement_dict[("Z",a_ind,self.time)] = self.meas_ind
            self.meas_ind += 1
        self.circuit.append("TICK")
        if basis == "X":
            Pstring = 'X'+str(self.qubit_dict[(0,0)])
            for q in [self.qubit_dict[(0,y)] for y in range(1,3*self.d_cc-2) if (0,y) in self.qubit_dict]:
                Pstring += '*X'+str(q)
            self.circuit.append("MPP",stim.PauliString(Pstring))
            self.meas_ind += 1
            self.circuit.append("OBSERVABLE_INCLUDE",[stim.target_rec(-1)],0)
        elif basis == "Z":
            Pstring = 'Z'+str(self.qubit_dict[(0,0)])
            for q in [self.qubit_dict[(xy,xy)] for xy in range(1,int(3*(self.d_cc-1)/2+1)) if (xy,xy) in self.qubit_dict]:
                Pstring += '*Z'+str(q)
            self.circuit.append("MPP",stim.PauliString(Pstring))
            self.meas_ind += 1
            self.circuit.append("OBSERVABLE_INCLUDE",[stim.target_rec(-1)],0)

        elif basis == "Y":
            Pstring = 'Y'+str(self.qubit_dict[(0,0)])
            for q in [self.qubit_dict[(xy,xy)] for xy in range(1,int(3*(self.d_cc-1)/2+1)) if (xy,xy) in self.qubit_dict]:
                Pstring += '*Z'+str(q)
            for q in [self.qubit_dict[(0,y)] for y in range(1,3*self.d_cc-2) if (0,y) in self.qubit_dict]:
                Pstring += '*X'+str(q)
            self.circuit.append("MPP",stim.PauliString(Pstring))
            self.meas_ind += 1
            self.circuit.append("OBSERVABLE_INCLUDE",[stim.target_rec(-1)],0)

        self.circuit.append("TICK")

        for coord, ind in self.qubit_dict.items():
            if coord[0]>coord[1] and coord[0]>=0:
                self.circuit.append("RX",ind)
        for coord, ind in self.qubit_dict.items():
            if coord[0]<0:
                self.circuit.append("R",ind)

        # self.circuit.append("TICK")

        if measure_sc_plaquettes:
            for plaq in self.sc_plaquettes:
                a_coord = plaq['ancilla_coord']
                a_ind = self.ancilla_dict[a_coord]
                q_inds = [self.qubit_dict[c] if c in self.qubit_dict else self.ancilla_dict[c] for c in plaq['qubits_coord']]

                if plaq in self.grafted_Xplaquettes and a_coord[0]%2 and a_coord[0]>0:
                    p_string = 'X'+str(q_inds[0])
                    for q in q_inds[1:]:
                        p_string += '*X'+str(q)
                    self.circuit.append("MPP",stim.PauliString(p_string))
                    self.measurement_dict[("X",a_ind,self.time)]=self.meas_ind
                    self.meas_ind+=1
                    self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("X",a_ind,self.time)]-self.meas_ind)],[1,a_ind,self.time])
                if plaq in self.grafted_Zplaquettes and a_coord[0]%2==0 and a_coord[0]<0:
                    p_string = 'Z'+str(q_inds[0])
                    for q in q_inds[1:]:
                        p_string += '*Z'+str(q)
                    self.circuit.append("MPP",stim.PauliString(p_string))
                    self.measurement_dict[("Z",a_ind,self.time)]=self.meas_ind
                    self.meas_ind+=1
                    self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("Z",a_ind,self.time)]-self.meas_ind)],[1,a_ind,self.time])
            self.circuit.append("TICK")

            self.time+=1

    def perfect_stab(self,Xstabs,Zstabs):

        for stab in Xstabs:
            q_inds = [self.qubit_dict[c] if c in self.qubit_dict else self.ancilla_dict[c] for c in stab['qubits_coord'] if c]
            a_ind = self.ancilla_dict[stab['ancilla_coord']]
            Pstring = 'X'+str(q_inds[0])
            for q in q_inds[1:]:
                Pstring += '*X'+str(q)
            self.circuit.append("MPP",stim.PauliString(Pstring))
            self.measurement_dict[("X",a_ind,self.time)] = self.meas_ind
            self.meas_ind += 1
            if ("X",a_ind,self.time-1) in self.measurement_dict:
                self.circuit.append("DETECTOR",[stim.target_rec(-1),stim.target_rec(self.measurement_dict[("X",a_ind,self.time-1)]-self.meas_ind)],[1,a_ind,self.time])

        self.circuit.append("TICK")

        for stab in Zstabs:
            q_inds = [self.qubit_dict[c] if c in self.qubit_dict else self.ancilla_dict[c] for c in stab['qubits_coord'] if c]
            a_ind = self.ancilla_dict[stab['ancilla_coord']]
            Pstring = 'Z'+str(q_inds[0])
            for q in q_inds[1:]:
                Pstring += '*Z'+str(q)
            self.circuit.append("MPP",stim.PauliString(Pstring))
            self.measurement_dict[("Z",a_ind,self.time)] = self.meas_ind
            self.meas_ind += 1
            if ("Z",a_ind,self.time-1) in self.measurement_dict:
                self.circuit.append("DETECTOR",[stim.target_rec(-1),stim.target_rec(self.measurement_dict[("Z",a_ind,self.time-1)]-self.meas_ind)],[0,a_ind,self.time])
        self.circuit.append("TICK")
        # self.circuit.append("TICK")

        self.time+=1

    def deform_to_sc(self, measure_sc_plaquettes=False):
        if self.d_cc>5:
            raise NotImplementedError()

        R_list = []
        RX_list = []
        trg_list_CX_list0 = [[] for _ in range(4)]
        trg_list_CX_list = [[] for _ in range(4)]
        MX1_list = []
        M1_list = []
        if measure_sc_plaquettes:
            for plaq in self.grafted_Xplaquettes:
                a_coord = plaq['ancilla_coord']
                if a_coord not in self.flag_qubit_coords:
                    a_ind = self.ancilla_dict[a_coord]
                    RX_list.append(a_ind)
                    for c in plaq['qubits_coord']:
                        q = self.qubit_dict[c]
                        if c==(a_coord[0],a_coord[1]+1):
                            trg_list_CX_list0[0]+=[a_ind,q]
                        elif c==(a_coord[0]+1,a_coord[1]):
                            trg_list_CX_list0[2]+=[a_ind,q]
                        elif c==(a_coord[0]-1,a_coord[1]):
                            trg_list_CX_list0[1]+=[a_ind,q]
                        elif c==(a_coord[0],a_coord[1]-1):
                            trg_list_CX_list0[3]+=[a_ind,q]
                    MX1_list.append(a_ind)
                    self.measurement_dict[("X",a_ind,self.time)]=self.meas_ind
                    self.meas_ind+=1
            for plaq in self.grafted_Zplaquettes:
                a_coord = plaq['ancilla_coord']
                if a_coord not in self.flag_qubit_coords and a_coord!=(2,1):
                    a_ind = self.ancilla_dict[a_coord]
                    R_list.append(a_ind)
                    for c in plaq['qubits_coord']:
                        q = self.qubit_dict[c]
                        if c==(a_coord[0],a_coord[1]+1):
                            trg_list_CX_list0[0]+=[q,a_ind]
                        elif c==(a_coord[0]+1,a_coord[1]):
                            trg_list_CX_list0[1]+=[q,a_ind]
                        elif c==(a_coord[0]-1,a_coord[1]):
                            trg_list_CX_list0[2]+=[q,a_ind]
                        elif c==(a_coord[0],a_coord[1]-1):
                            trg_list_CX_list0[3]+=[q,a_ind]

                    M1_list.append(a_ind)
                    self.measurement_dict[("Z",a_ind,self.time)]=self.meas_ind
                    self.meas_ind+=1
            self.circuit.append("TICK")

        for plaq in self.cc_plaquettes:
            RX_list += [self.ancilla_dict[plaq["ancilla1_coord"]]]
            R_list += [self.ancilla_dict[plaq["ancilla2_coord"]]]
            trg_list_CX_list[0] += [self.ancilla_dict[plaq["ancilla1_coord"]],self.ancilla_dict[plaq["ancilla2_coord"]]]

        if (3*(self.d_cc-1)/2,3*(self.d_cc-1)/2+1) in self.ancilla_dict:
            boundary_Zanc_ind = self.ancilla_dict[(3*(self.d_cc-1)/2,3*(self.d_cc-1)/2+1)]
            q1_ind = self.qubit_dict[(3*(self.d_cc-1)/2,3*(self.d_cc-1)/2)]
            q2_ind = self.qubit_dict[(3*(self.d_cc-1)/2-1,3*(self.d_cc-1)/2+1)]
            R_list+=[boundary_Zanc_ind]
            trg_list_CX_list[0]+=[q1_ind,boundary_Zanc_ind]
            trg_list_CX_list[3]+=[q2_ind,boundary_Zanc_ind]
        else:
            boundary_Zanc_ind = None

        self.circuit.append("R",R_list)
        self.circuit.append("RX",RX_list)
        self.circuit.append("TICK")

        for plaq in self.cc_plaquettes:
            anc1_coord, anc2_coord = plaq['ancilla1_coord'],plaq['ancilla2_coord']
            anc1, anc2 = self.ancilla_dict[anc1_coord],self.ancilla_dict[anc2_coord]
            flag = self.ancilla_dict[plaq['flag_coord']]
            qubits_1 = [self.qubit_dict[coord] if coord else None for coord in plaq['qubits1_coord']] 
            qubits_2 = [self.qubit_dict[coord] if coord else None for coord in plaq['qubits2_coord']]

            for i,q1 in enumerate(qubits_1,start=1):
                if q1!=None:
                    if (anc1_coord[1]%6==1 and i<3) or (anc1_coord[1]%6==4 and i==3):
                        if (anc1_coord==(0,1) and i==2):
                            trg_list_CX_list[i]+=[self.ancilla_dict[(2,1)],self.qubit_dict[(1,1)]]
                        else:
                            trg_list_CX_list[i]+=[anc1,q1]
                    else:
                        if (anc1_coord==(2,1)):
                            trg_list_CX_list[i]+=[self.qubit_dict[(2,0)],self.ancilla_dict[(2,1)]]
                        else:
                            trg_list_CX_list[i]+=[q1,anc1]
            for i,q2 in enumerate(qubits_2,start=1):
                if q2!=None:
                    if (anc1_coord[1]%6==1 and i<3) or (anc1_coord[1]%6==4 and i==3):
                        trg_list_CX_list[i]+=[anc2,q2]
                    else:
                        if (anc2_coord==(2,3)):
                            trg_list_CX_list[i]+=[self.ancilla_dict[(2,3)],self.qubit_dict[(1,3)]]
                        else:
                            trg_list_CX_list[i]+=[q2,anc2]
        
        trg_list_CX_list.append([self.ancilla_dict[(0,1)],self.qubit_dict[(1,1)],
                                 self.ancilla_dict[(0,3)],self.qubit_dict[(1,3)],
                                 self.qubit_dict[(3,1)],self.ancilla_dict[(2,1)],
                                 self.ancilla_dict[(2,3)],self.qubit_dict[(3,3)]])
        trg_list_CX_list.append([self.qubit_dict[(1,1)],self.ancilla_dict[(2,1)],
                                 self.qubit_dict[(1,3)],self.ancilla_dict[(0,3)],
                                 self.qubit_dict[(3,3)],self.ancilla_dict[(2,3)]])

        for CX_list in trg_list_CX_list0:
            if CX_list:
                self.circuit.append("CX",CX_list)
                self.circuit.append("TICK")

        for CX_list in trg_list_CX_list:
            self.circuit.append("CX",CX_list)
            self.circuit.append("TICK")

        self.circuit.append("MX",MX1_list)
        self.circuit.append("M",M1_list)


        if measure_sc_plaquettes:
            for plaq in self.grafted_Xplaquettes:
                a_coord = plaq['ancilla_coord']
                if a_coord not in self.flag_qubit_coords:
                    a_ind = self.ancilla_dict[a_coord]
                    if a_coord[0]>0:
                        self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("X",a_ind,self.time)]-self.meas_ind)],[1,a_ind,self.time])
            for plaq in self.grafted_Zplaquettes:
                a_coord = plaq['ancilla_coord']
                if a_coord not in self.flag_qubit_coords and a_coord!=(2,1):
                    a_ind = self.ancilla_dict[a_coord]
                    if a_coord[0]<0:
                        self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("Z",a_ind,self.time)]-self.meas_ind)],[0,a_ind,self.time])
            self.time+=1


        MX_list = []
        for plaq in self.cc_plaquettes:
            anc1_coord, anc2_coord = plaq['ancilla1_coord'],plaq['ancilla2_coord']
            anc1, anc2 = self.ancilla_dict[anc1_coord],self.ancilla_dict[anc2_coord]
            flag = self.ancilla_dict[plaq['flag_coord']]
            if anc1_coord[1]%6==1:
                if anc1_coord[1]!=1:
                    MX_list+=[anc1,anc2]
                    anc1_include = (anc1_coord[0]-1,anc1_coord[1]-2)
                    if anc1_include in self.ancilla_dict:
                        self.measurement_dict[("X",self.ancilla_dict[anc1_include],self.time-0.5)] = self.meas_ind
                    anc2_include = (anc2_coord[0]-1,anc2_coord[1]+2)
                    if anc2_include in self.ancilla_dict:
                        self.measurement_dict[("X",self.ancilla_dict[anc2_include],self.time+0.5)] = self.meas_ind+1
                    if anc1_coord[0]==0:
                        self.measurement_dict[("X",anc1,self.time)] = self.meas_ind
                        self.measurement_dict[("X",anc2,self.time)] = self.meas_ind+1
                    self.meas_ind+=2
                else:
                    MX_list+=[anc2]
                    anc2_include = (anc2_coord[0]+1,anc2_coord[1]+2)
                    if anc2_include in self.ancilla_dict:
                        self.measurement_dict[("X",self.ancilla_dict[anc2_include],self.time+0.5)] = self.meas_ind
                    if anc1_coord[0]==0:
                        self.measurement_dict[("X",anc2,self.time)] = self.meas_ind
                    self.meas_ind+=1
        self.circuit.append("MX",MX_list)
        M_list = []
        if boundary_Zanc_ind:
            M_list += [boundary_Zanc_ind]
            self.measurement_dict[("Z",boundary_Zanc_ind,self.time-1)] = self.meas_ind
            self.meas_ind+=1
        for plaq in self.cc_plaquettes:
            anc1_coord, anc2_coord = plaq['ancilla1_coord'],plaq['ancilla2_coord']
            anc1, anc2 = self.ancilla_dict[anc1_coord],self.ancilla_dict[anc2_coord]
            flag = self.ancilla_dict[plaq['flag_coord']]
            if anc1_coord[1]%6==4:
                M_list+=[anc1,anc2]
                anc1_include = (anc1_coord[0]-1,anc1_coord[1]-2)
                if anc1_include in self.ancilla_dict:
                    self.measurement_dict[("Z",self.ancilla_dict[anc1_include],self.time-0.5)] = self.meas_ind
                anc2_include = (anc2_coord[0]-1,anc2_coord[1]+2)
                if anc2_include in self.ancilla_dict:
                    self.measurement_dict[("Z",self.ancilla_dict[anc2_include],self.time+0.5)] = self.meas_ind+1
                anc3_include = (anc1_coord[0]-1,anc1_coord[1]-1)
                if anc3_include in self.ancilla_dict and anc3_include[0] == anc3_include[1]+1:
                    self.measurement_dict[("Z",self.ancilla_dict[anc3_include],self.time-0.5)] = self.meas_ind
                self.meas_ind+=2
        M_list+=[self.ancilla_dict[(2,1)]]
        self.measurement_dict[("Z",self.ancilla_dict[(2,1)],self.time+0.3)] = self.meas_ind
        self.meas_ind+=1
        self.circuit.append("M",M_list) #self.meas_ind updated above
        if ("Z",self.ancilla_dict[(2,1)],self.time-1) in self.measurement_dict:
            self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("Z",self.ancilla_dict[(2,1)],self.time-1)]-self.meas_ind),
                                            stim.target_rec(-1)],
                                            [0,self.ancilla_dict[(2,1)],self.time])
        else:
            self.measurement_dict[("Z",self.ancilla_dict[(2,1)],self.time-1)] = self.meas_ind-1
        self.circuit.append("TICK")

        R2_list = []
        RX2_list = []
        for plaq in self.cc_plaquettes:
            anc1_coord, anc2_coord = plaq['ancilla1_coord'],plaq['ancilla2_coord']
            flag = self.ancilla_dict[plaq['flag_coord']]
            if anc1_coord[1]%6==1:
                RX2_list+=[flag]
            elif anc1_coord[1]%6==4:
                R2_list+=[flag]
        self.circuit.append("R",R2_list)
        self.circuit.append("RX",RX2_list)


        ### handling special cases
        left_qubits, mid_qubits, right_qubits = [], [], []
        left_ancs, right_ancs, Xdet_ancs, Zdet_ancs = [], [], [], []
        for offset in range(0,int(3*(self.d_cc-1)/2),6):
            left_qubits.append(self.qubit_dict[(2+offset,3*self.d_cc-5-offset)])
            mid_qubits.append(self.ancilla_dict[(1+offset,3*self.d_cc-4-offset)])
            right_qubits.append(self.qubit_dict[(0+offset,3*self.d_cc-3-offset)])
            left_ancs.append(self.ancilla_dict[(2+offset,3*self.d_cc-4-offset)])
            right_ancs.append(self.ancilla_dict[(1+offset,3*self.d_cc-3-offset)])
            Xdet_ancs.append(self.ancilla_dict[(1+offset,3*self.d_cc-4-offset)])
            Zdet_ancs.append(self.ancilla_dict[(1+offset,3*self.d_cc-4-offset)])

        self.circuit.append("R",left_ancs+[self.ancilla_dict[(2,1)]])
        self.circuit.append("RX",right_ancs)
        self.circuit.append("TICK")
        self.circuit.append("CX",[q for left_qubit,left_anc,right_anc,right_qubit in zip(left_qubits,left_ancs,right_ancs,right_qubits) for q in [left_qubit,left_anc,right_anc,right_qubit]])
        self.circuit.append("CX",[self.ancilla_dict[(0,1)],self.qubit_dict[(1,1)]])
        self.circuit.append("TICK")
        self.circuit.append("CX",[q for left_anc, mid_qubit in zip(left_ancs,mid_qubits) for q in [left_anc,mid_qubit]])
        self.circuit.append("CX",[self.qubit_dict[(1,1)],self.ancilla_dict[(2,1)]])
        self.circuit.append("TICK")
        self.circuit.append("CX",[q for mid_qubit, right_anc in zip(mid_qubits,right_ancs) for q in [mid_qubit,right_anc]])
        self.circuit.append("TICK")
        self.circuit.append("MX",left_ancs)
        for Xdet_anc in Xdet_ancs:
            self.measurement_dict[("X",Xdet_anc,self.time-0.5)]=self.meas_ind
            self.meas_ind+=1
        self.circuit.append("M",right_ancs)
        for Zdet_anc in Zdet_ancs:
            self.measurement_dict[("Z",Zdet_anc,self.time-0.5)]=self.meas_ind
            self.meas_ind+=1
        self.circuit.append("M",self.ancilla_dict[(2,1)])
        self.measurement_dict[("Z",self.ancilla_dict[(2,2)],self.time+0.5)]=self.meas_ind
        self.meas_ind+=1
        self.circuit.append("MX",self.ancilla_dict[(0,1)])
        self.measurement_dict[("X",self.ancilla_dict[(0,1)],self.time+0.5)]=self.meas_ind
        self.meas_ind+=1

        #surface code stabilizer measurements
        self.circuit.append("TICK")
        # self.circuit.append("TICK")

        if measure_sc_plaquettes:
            for plaq in self.sc_plaquettes:
                a_coord = plaq['ancilla_coord']
                a_ind = self.ancilla_dict[a_coord]
                q_inds = [self.qubit_dict[c] if c in self.qubit_dict else self.ancilla_dict[c] for c in plaq['qubits_coord']]
                if a_coord[0]%2:
                    p_string = 'X'+str(q_inds[0])
                    for q in q_inds[1:]:
                        p_string += '*X'+str(q)
                    self.circuit.append("MPP",stim.PauliString(p_string))
                    self.measurement_dict[("X",a_ind,self.time)]=self.meas_ind
                    self.meas_ind+=1
                else:
                    p_string = 'Z'+str(q_inds[0])
                    for q in q_inds[1:]:
                        p_string += '*Z'+str(q)
                    self.circuit.append("MPP",stim.PauliString(p_string))
                    self.measurement_dict[("Z",a_ind,self.time)]=self.meas_ind
                    self.meas_ind+=1
        else:
            for plaq in self.sc_plaquettes:
                a_coord = plaq['ancilla_coord']
                a_ind = self.ancilla_dict[a_coord]
                q_inds = [self.qubit_dict[c] if c in self.qubit_dict else self.ancilla_dict[c] for c in plaq['qubits_coord']]
                if a_coord[0]%2:
                    p_string = 'X'+str(q_inds[0])
                    for q in q_inds[1:]:
                        p_string += '*X'+str(q)
                    self.circuit.append("MPP",stim.PauliString(p_string))
                    self.measurement_dict[("X",a_ind,self.time)]=self.meas_ind
                    self.meas_ind+=1
                else:
                    p_string = 'Z'+str(q_inds[0])
                    for q in q_inds[1:]:
                        p_string += '*Z'+str(q)
                    self.circuit.append("MPP",stim.PauliString(p_string))
                    self.measurement_dict[("Z",a_ind,self.time)]=self.meas_ind
                    self.meas_ind+=1

        for plaq in self.sc_plaquettes:
            a_coord = plaq['ancilla_coord']
            a_ind = self.ancilla_dict[a_coord]
            q_inds = [self.qubit_dict[c] if c in self.qubit_dict else self.ancilla_dict[c] for c in plaq['qubits_coord']]
            a_above_ind = self.ancilla_dict[(a_coord[0]+1,a_coord[1])] if (a_coord[0]+1,a_coord[1]) in self.ancilla_dict else -1
            a_right_ind = self.ancilla_dict[(a_coord[0],a_coord[1]+1)] if (a_coord[0],a_coord[1]+1) in self.ancilla_dict else -1
            a_right2_ind = self.ancilla_dict[(a_coord[0],a_coord[1]+2)] if (a_coord[0],a_coord[1]+2) in self.ancilla_dict else -1

            if a_coord[0]%2:
                # handle unchanged SC detectors
                if plaq in self.grafted_Xplaquettes:
                    self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("X",a_ind,t)]-self.meas_ind) for t in [self.time-1,self.time]],[1,a_ind,self.time])
                    
                # handle CC detectors that deform into a pair of SC stabilizers
                elif a_right_ind in [self.ancilla_dict[p['flag_coord']] for p in self.cc_plaquettes]:
                    self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("X",a_right_ind,t)]-self.meas_ind)
                                                    for t in [self.time-1,self.time-0.5,self.time+0.5]
                                                    if ("X",a_right_ind,t) in self.measurement_dict]+
                                                    [stim.target_rec(self.measurement_dict[("X",a_ind,self.time)]-self.meas_ind)]+
                                                    ([stim.target_rec(self.measurement_dict[("X",a_right2_ind,self.time)]-self.meas_ind)] if ("X",a_right2_ind,self.time) in self.measurement_dict else [])
                                                    ,[1,a_ind,self.time])
                    
                # handle CC detectors that deform into a single SC stabilizer
                elif a_above_ind in [self.ancilla_dict[p['flag_coord']] for p in self.cc_plaquettes] and a_coord!=(1,2):
                    if a_coord[0]>0:
                        self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("X",a_above_ind,t)]-self.meas_ind)
                                                        for t in [self.time-1,self.time-0.5,self.time+0.5]
                                                        if ("X",a_above_ind,t) in self.measurement_dict]+
                                                        [stim.target_rec(self.measurement_dict[("X",a_ind,self.time)]-self.meas_ind)]
                                                        ,[1,a_ind,self.time])
                    else:
                        #triangles at base
                        self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("X",a_ind,self.time-1)]-self.meas_ind),
                                                        stim.target_rec(self.measurement_dict[("X",a_ind,self.time)]-self.meas_ind)]
                                                        ,[1,a_ind,self.time])
                        #CC detectors that are discontinued
                        if a_coord!=(-1,2):
                            a_above_left_ind = self.ancilla_dict[(a_coord[0]+1,a_coord[1]-1)]
                            a_above_right_ind = self.ancilla_dict[(a_coord[0]+1,a_coord[1]+1)]
                            self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("X",a_above_ind,self.time-1)]-self.meas_ind),
                                                            stim.target_rec(self.measurement_dict[("X",a_above_left_ind,self.time)]-self.meas_ind),
                                                            stim.target_rec(self.measurement_dict[("X",a_above_right_ind,self.time)]-self.meas_ind)]
                                                            ,[1,a_ind,self.time])
                elif a_coord==(3,2):
                    self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(a_coord[0]-1,a_coord[1])],self.time-1)]-self.meas_ind),
                                                    stim.target_rec(self.measurement_dict[("X",a_ind,self.time)]-self.meas_ind)]
                                                    ,[1,a_ind,self.time])
                elif a_coord==(1,0):
                    self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[label]-self.meas_ind) for label in [("X",self.ancilla_dict[(0,2)],self.time-1),
                                                                                                                              ("X",self.ancilla_dict[(1,2)],self.time),
                                                                                                                              ("X",self.ancilla_dict[(0,1)],self.time+0.5),
                                                                                                                              ("X",a_ind,self.time)]]
                                                    ,[1,a_ind,self.time])

            else:
                # handle unchanged SC detectors
                if plaq in self.grafted_Zplaquettes:
                    self.circuit.append("DETECTOR", [stim.target_rec(self.measurement_dict[("Z",a_ind,t)]-self.meas_ind)
                                                     for t in [self.time-1,self.time-0.5,self.time+0.5,self.time]
                                                     if ("Z",a_ind,t) in self.measurement_dict]
                                                    ,[0,a_ind,self.time])

                elif a_coord[0]==a_coord[1]+1>2:
                    #traingles at left edge
                    self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("Z",a_ind,self.time-1)]-self.meas_ind),stim.target_rec(self.measurement_dict[("Z",a_ind,self.time)]-self.meas_ind)]
                                                    ,[0,a_ind,self.time])
                    
                # handle CC detectors that deform into a pair of SC stabilizers
                elif a_right_ind in [self.ancilla_dict[p['flag_coord']] for p in self.cc_plaquettes]:
                    if a_right_ind==self.ancilla_dict[(2,2)]:
                        self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("Z",a_right_ind,t)]-self.meas_ind)
                                                        for t in [self.time-1,self.time-0.5,self.time+0.5]
                                                        if ("Z",a_right_ind,t) in self.measurement_dict]+
                                                        [stim.target_rec(self.measurement_dict[("Z",self.ancilla_dict[(2,1)],self.time+0.3)]-self.meas_ind)]+
                                                        [stim.target_rec(self.measurement_dict[("Z",a_ind,self.time)]-self.meas_ind)]+
                                                        [stim.target_rec(self.measurement_dict[("Z",a_right2_ind,self.time)]-self.meas_ind)] if ("Z",a_right2_ind,self.time) in self.measurement_dict else []
                                                        ,[0,a_ind,self.time])
                    else:
                        self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("Z",a_right_ind,t)]-self.meas_ind)
                                                        for t in [self.time-1,self.time-0.5,self.time+0.5]
                                                        if ("Z",a_right_ind,t) in self.measurement_dict]+
                                                        [stim.target_rec(self.measurement_dict[("Z",a_ind,self.time)]-self.meas_ind)]+
                                                        [stim.target_rec(self.measurement_dict[("Z",a_right2_ind,self.time)]-self.meas_ind)] if ("Z",a_right2_ind,self.time) in self.measurement_dict else []
                                                        ,[0,a_ind,self.time])

                # handle CC detectors that deform into a single SC stabilizer
                elif a_above_ind in [self.ancilla_dict[p['flag_coord']] for p in self.cc_plaquettes]:
                    if a_above_ind in Zdet_ancs:
                        self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("Z",a_above_ind,t)]-self.meas_ind)
                                                        for t in [self.time-1,self.time-0.5,self.time+0.5]
                                                        if ("Z",a_above_ind,t) in self.measurement_dict]+
                                                        [stim.target_rec(self.measurement_dict[("Z",a_ind,self.time)]-self.meas_ind)]+
                                                        [stim.target_rec(self.measurement_dict[("Z",self.ancilla_dict[a_coord[0]+2,a_coord[1]],self.time)]-self.meas_ind)]
                                                        ,[0,a_ind,self.time])
                    else:
                        self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("Z",a_above_ind,t)]-self.meas_ind)
                                                        for t in [self.time-1,self.time-0.5,self.time+0.5]
                                                        if ("Z",a_above_ind,t) in self.measurement_dict]+
                                                        [stim.target_rec(self.measurement_dict[("Z",a_ind,self.time)]-self.meas_ind)]
                                                        ,[0,a_ind,self.time])
                elif boundary_Zanc_ind and a_ind==boundary_Zanc_ind:
                    self.circuit.append("DETECTOR",[stim.target_rec(self.measurement_dict[("Z",a_ind,self.time-1)]-self.meas_ind),stim.target_rec(self.measurement_dict[("Z",a_ind,self.time)]-self.meas_ind)]
                                                    ,[0,a_ind,self.time])

        self.circuit.append("TICK")
        self.circuit.append("TICK")

        self.time+=1

    def stab(self, add_detectors = True):
        d = self.d_cc
        _qubit_dict, _ancilla_dict, _, _, boundary_qubit_coords, plaquettes, _, _, _ = qubit_coord_groups_escape(d)

        pipeline_with_previous = self.pipeline

        #Bell-state prep
        trg_list_R1 = []
        trg_list_R2 = []
        trg_list_H = []
        trg_list_M = []
        trg_list_M1 = []
        trg_list_M2 = []
        trg_list_CX_list = [[] for _ in range(10)]
        for plaquette in plaquettes:
            anc1, anc2, flag = self.ancilla_dict[plaquette['ancilla1_coord']],self.ancilla_dict[plaquette['ancilla2_coord']],self.ancilla_dict[plaquette['flag_coord']]
            trg_list_H+=[flag]
            trg_list_M+=[anc1, anc2, flag]
            if plaquette['flag_coord'][1]%6==2: #early shift
                trg_list_R1+=[anc2, flag]
                trg_list_R2+=[anc1]
                trg_list_M2+=[anc1, flag]
                trg_list_M1+=[anc2]
                trg_list_CX_list[0]+=[flag, anc2]
                trg_list_CX_list[1]+=[flag,anc1]
                trg_list_CX_list[2]+=[anc1,flag]
                trg_list_CX_list[-1]+=[flag, anc1]
                trg_list_CX_list[-2]+=[flag,anc2]
                trg_list_CX_list[-3]+=[anc2,flag]
            else:
                trg_list_R1+=[anc1, flag]
                trg_list_R2+=[anc2]
                trg_list_M2+=[anc2, flag]
                trg_list_M1+=[anc1]
                trg_list_CX_list[0]+=[flag, anc1]
                trg_list_CX_list[1]+=[flag,anc2]
                trg_list_CX_list[2]+=[anc2,flag]
                trg_list_CX_list[-1]+=[flag, anc2]
                trg_list_CX_list[-2]+=[flag,anc1]
                trg_list_CX_list[-3]+=[anc1,flag]

        #X map
        for plaquette in plaquettes:
            anc1, anc2 = self.ancilla_dict[plaquette['ancilla1_coord']],self.ancilla_dict[plaquette['ancilla2_coord']]
            qubits_1 = [self.qubit_dict[coord] if coord else None for coord in plaquette['qubits1_coord']] 
            qubits_2 = [self.qubit_dict[coord] if coord else None for coord in plaquette['qubits2_coord']]

            if plaquette['flag_coord'][1]%6==2: #anc2 early shift
                for i,q1 in enumerate(qubits_1,start=3):
                    if q1!=None:
                        trg_list_CX_list[i]+=[anc1,q1]
                for i,q2 in enumerate(qubits_2,start=1):#1
                    if q2!=None:
                        trg_list_CX_list[i]+=[anc2,q2]
            else:
                for i,q1 in enumerate(qubits_1,start=1):#1
                    if q1!=None:
                        trg_list_CX_list[i]+=[anc1,q1]
                for i,q2 in enumerate(qubits_2,start=3):
                    if q2!=None:
                        trg_list_CX_list[i]+=[anc2,q2]

        #Z map
        for plaquette in plaquettes:
            anc1, anc2 = self.ancilla_dict[plaquette['ancilla1_coord']],self.ancilla_dict[plaquette['ancilla2_coord']]
            qubits_1 = [self.qubit_dict[coord] if coord else None for coord in plaquette['qubits1_coord']] 
            qubits_2 = [self.qubit_dict[coord] if coord else None for coord in plaquette['qubits2_coord']]
            if plaquette['flag_coord'][1]%6==2: #anc2 early shift
                for i,q1 in enumerate(qubits_1,start=6):
                    if q1!=None:
                        trg_list_CX_list[i]+=[q1,anc1]
                for i,q2 in enumerate(qubits_2,start=4):#3
                    if q2:
                        trg_list_CX_list[i]+=[q2,anc2]
            else:
                for i,q1 in enumerate(qubits_1,start=4):#3
                    if q1!=None:
                        trg_list_CX_list[i]+=[q1,anc1]
                for i,q2 in enumerate(qubits_2,start=6):
                    if q2!=None:
                        trg_list_CX_list[i]+=[q2,anc2]


        if not pipeline_with_previous:
            self.circuit.append("TICK")
            ##we cannot put the reset before the classically controlled Paulis... (stim issue?)
        if self.no_hadamards:
            self.circuit.append("R",set(trg_list_R1)-set(trg_list_H))
            self.circuit.append("RX",trg_list_H)
        else:
            self.circuit.append("R",trg_list_R1)
            self.circuit.append("H",trg_list_H)
        if pipeline_with_previous:
            self.circuit.append("TICK")


        self.circuit.append("R",trg_list_R2)
        if not pipeline_with_previous:
            self.circuit.append("TICK")
        self.circuit.append("CX",trg_list_CX_list[0])
        self.circuit.append("TICK")

        for trgs in trg_list_CX_list[1:-1]:
            self.circuit.append("CX",trgs)
            self.circuit.append("TICK")

        #Bell-state meas
        self.circuit.append("CX",trg_list_CX_list[-1])
        if not pipeline_with_previous:
            self.circuit.append("TICK")
        self.circuit.append("M",trg_list_M1)
        if pipeline_with_previous:
            self.circuit.append("TICK")
        for plaquette in plaquettes:
            # in the measurement_dict we keep the labelling as it was in the uncompressed circuit
            anc1_coord = plaquette['ancilla1_coord']
            anc1, anc2, flag = self.ancilla_dict[plaquette['ancilla1_coord']],self.ancilla_dict[plaquette['ancilla2_coord']],self.ancilla_dict[plaquette['flag_coord']]
            if anc1 in trg_list_M1 or anc2 in trg_list_M1:
                self.measurement_dict[("f",flag,self.time)] = self.meas_ind
                self.meas_ind+=1

        #Bell-state meas
        if self.no_hadamards:
            for trg in trg_list_M2:
                if trg in trg_list_H:
                    self.circuit.append("MX",trg)
                else:
                    self.circuit.append("M",trg)
            for plaquette in plaquettes:
                # in the measurement_dict we keep the labelling as it was in the uncompressed circuit
                anc1_coord = plaquette['ancilla1_coord']
                anc1, anc2, flag = self.ancilla_dict[plaquette['ancilla1_coord']],self.ancilla_dict[plaquette['ancilla2_coord']],self.ancilla_dict[plaquette['flag_coord']]
                if flag in trg_list_M2:                    
                    self.measurement_dict[("Z",anc2,self.time)] = self.meas_ind
                    self.measurement_dict[("X",anc1,self.time)] = self.meas_ind + 1
                    self.meas_ind+=2
        else:
            self.circuit.append("H",trg_list_H)
            self.circuit.append("M",trg_list_M2)
            for plaquette in plaquettes:
                # in the measurement_dict we keep the labelling as it was in the uncompressed circuit
                anc1_coord = plaquette['ancilla1_coord']
                anc1, anc2, flag = self.ancilla_dict[plaquette['ancilla1_coord']],self.ancilla_dict[plaquette['ancilla2_coord']],self.ancilla_dict[plaquette['flag_coord']]
                if flag in trg_list_M2:                    
                    self.measurement_dict[("Z",anc2,self.time)] = self.meas_ind
                    self.measurement_dict[("X",anc1,self.time)] = self.meas_ind + 1
                    self.meas_ind+=2

        if not pipeline_with_previous:
            self.circuit.append("TICK")


        for plaquette in plaquettes:
            anc1_coord = plaquette['ancilla1_coord']
            anc1, anc2, flag = self.ancilla_dict[plaquette['ancilla1_coord']],self.ancilla_dict[plaquette['ancilla2_coord']],self.ancilla_dict[plaquette['flag_coord']]
            if add_detectors:
                ## flag detectors
                self.circuit.append("DETECTOR",
                                    [stim.target_rec(self.measurement_dict[("f",flag,self.time)]-self.meas_ind)],
                                    [])
                ## Z detectors
                self.circuit.append("DETECTOR",
                                    [stim.target_rec(self.measurement_dict[("Z",anc2,self.time)]-self.meas_ind)],
                                    [0,anc1,self.time])
                ## X detectors
                if self.time>0:
                    # stabilizers along the base
                    if anc1_coord[0]==0:
                        self.circuit.append("DETECTOR",
                                            [stim.target_rec(self.measurement_dict[("X",anc1,self.time)]-self.meas_ind),
                                            stim.target_rec(self.measurement_dict[("X",anc1,self.time-1)]-self.meas_ind),
                                            stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(anc1_coord[0]+2,anc1_coord[1])],self.time-1)]-self.meas_ind)],
                                            [1,anc1,self.time])

                    # stabilizers along the legs
                    elif (anc1_coord[0],anc1_coord[1]-1) not in _qubit_dict or (anc1_coord[0],anc1_coord[1]+3) not in _qubit_dict: 
                        # the flag above exists (the ancilla could exist due to the link)
                        if (anc1_coord[0]-2,anc1_coord[1]+1) in _ancilla_dict: 
                            self.circuit.append("DETECTOR",
                                                [stim.target_rec(self.measurement_dict[("X",anc1,self.time)]-self.meas_ind),
                                                stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(anc1_coord[0]-2,anc1_coord[1])],self.time-1)]-self.meas_ind)],
                                                [1,anc1,self.time])
                        else:
                            self.circuit.append("DETECTOR",
                                                [stim.target_rec(self.measurement_dict[("X",anc1,self.time)]-self.meas_ind)],
                                                [1,anc1,self.time])
                    # stabilizers in the bulk
                    else:
                        if (anc1_coord[0]-2,anc1_coord[1]+1) in _ancilla_dict: 
                            self.circuit.append("DETECTOR",
                                                [stim.target_rec(self.measurement_dict[("X",anc1,self.time)]-self.meas_ind),
                                                stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(anc1_coord[0]-2,anc1_coord[1])],self.time-1)]-self.meas_ind),
                                                stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(anc1_coord[0]+2,anc1_coord[1])],self.time-1)]-self.meas_ind)],
                                                [1,anc1,self.time])
                        else:
                            self.circuit.append("DETECTOR",
                                                [stim.target_rec(self.measurement_dict[("X",anc1,self.time)]-self.meas_ind),
                                                stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(anc1_coord[0]+2,anc1_coord[1])],self.time-1)]-self.meas_ind)],
                                                [1,anc1,self.time])

        if add_detectors:
            logical_support = set(_qubit_dict.keys())-set(boundary_qubit_coords)
            ancilla_coords_in_logical = [plaquette['ancilla1_coord'] for plaquette in plaquettes if len(logical_support & set(plaquette['qubits1_coord']))%2]
            if (any([("log_datX",self.qubit_dict[(-1,1)],t) in self.measurement_dict for t in range(self.time+1)]) or 
                any([("log_datY",self.qubit_dict[(-1,1)],t) in self.measurement_dict for t in range(self.time+1)])):
                self.circuit.append("OBSERVABLE_INCLUDE",
                                        [stim.target_rec(
                                            self.measurement_dict[("X",self.ancilla_dict[coord],self.time)] - self.meas_ind) 
                                            for coord in ancilla_coords_in_logical]
                                        ,0)

        self.time+=1

    def noiseless_final_meas(self, d = None, basis='Z', direct_grow = False):
        if basis=='Z':
            log_inds = self.sc_logical_Z
            pauli_string = 'Z'+str(log_inds[0])
            for q in log_inds[1:]:
                pauli_string+="*"+'Z'+str(q)
        elif basis =='X':
            log_inds = self.sc_logical_X
            pauli_string = 'X'+str(log_inds[0])
            for q in log_inds[1:]:
                pauli_string+="*"+'X'+str(q)
        elif basis =='Y':
            pauli_string = 'X'+str(self.sc_logical_X[0])
            for q in self.sc_logical_X[1:-1]:
                pauli_string+="*"+'X'+str(q)
            pauli_string+="*"+'Y'+str(self.sc_logical_X[-1])
            for q in self.sc_logical_Z[:-1]:
                pauli_string+="*"+'Z'+str(q)

        observable_includes = []

        if self.grow:
            if basis=="Z" or basis=="Y":
                for plaq in self.sc_plaquettes:
                    a_coord = plaq["ancilla_coord"]
                    if a_coord[0]%2==0 and a_coord[0]>a_coord[1] and a_coord[0]>=0:
                        observable_includes+=[self.measurement_dict[("Z",self.ancilla_dict[a_coord],1-direct_grow)]]
            if basis=="X" or basis=="Y":
                for plaq in self.sc_plaquettes:
                    a_coord = plaq["ancilla_coord"]
                    if a_coord[0]%2 and a_coord[0]<=0:
                        observable_includes+=[self.measurement_dict[("X",self.ancilla_dict[a_coord],1-direct_grow)]]


        self.circuit.append("MPP",[stim.PauliString(pauli_string)])
        self.measurement_dict[("final_log_meas",self.time)] = self.meas_ind
        observable_includes += [self.meas_ind]
        self.meas_ind+=1
        self.circuit.append("OBSERVABLE_INCLUDE",[stim.target_rec(i-self.meas_ind) for i in observable_includes],0)
