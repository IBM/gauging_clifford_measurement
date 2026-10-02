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
from utils import qubit_coord_groups

class ColorCode:
    """version 2.0 with pipelining"""
    def __init__(self, max_d):
        self.max_d = max_d

        self.no_hadamards = True
        self.pipeline = True


        self.qubit_dict, self.ancilla_dict, _, _, _, plaquettes = qubit_coord_groups(max_d)
        self.every_qubit_set = set(list(self.qubit_dict.values())+list(self.ancilla_dict.values()))
            
        self.circuit = stim.Circuit()
        for coord, q in self.qubit_dict.items():
            self.circuit.append("QUBIT_COORDS",q,[3*max_d-(3*max_d-2-coord[1])-3,coord[0]+1])
        for coord, q in self.ancilla_dict.items():
            self.circuit.append("QUBIT_COORDS",q,[3*max_d-(3*max_d-2-coord[1])-3,coord[0]+1])

        self.stim_polygon_comments = ""
        for plaquette in plaquettes:
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
    
    def Bell_prep(self, d = None, basis = "Y"):
        if not d:
            d = self.max_d
        _, _, _, _, _, _ = qubit_coord_groups(d)
        self.circuit.append("R",[0])
        if basis == "X":
            self.circuit.append("H",[0])
        if basis == "Y":
            self.circuit.append("H",[0])
            self.circuit.append("S",[0])
        self.grow(1, d)

        self.circuit.append("MPAD",0)
        self.meas_ind+=1
        self.measurement_dict[("log_dat"+basis,0,self.time-1)] = self.meas_ind-1
        self.last_logmeas_time = self.time-1

    def unitary_prep(self, d = None, basis = "Y"):
        assert d == 3
        assert basis == "Y"

        pipeline = self.pipeline

        every_qubit_dict = self.ancilla_dict.copy()
        every_qubit_dict.update(self.qubit_dict.copy())

        # from arXiv:2409.17595
        trg_list_R = [every_qubit_dict[coord] 
                      for coord in [(0, 4),(0, 1),(0, 2),(0, 3),(0, 5),
                                    (1, 3),(1, 2),(1, 4),(1, 5),
                                    (2, 4),(2, 2),(2, 3),
                                    (3, 2)]
                                    ]
        trg_list_H = [every_qubit_dict[coord] 
                      for coord in [(0, 4),(0, 1),(0, 2),(0, 5),
                                    (1, 3),(1, 2),(1, 5),
                                    (2, 4),(2, 2),
                                    (3, 2)]
                                    ]
        trg_list_CX1 = [every_qubit_dict[coord] 
                        for coord in [(0,4),(0,3),
                                      (1,3),(1,4),
                                      (2,4),(2,3)]
                                      ]
        trg_list_CX2 = [every_qubit_dict[coord] 
                        for coord in [(0,4),(1, 4),
                                      (1,3),(2, 3)]
                                      ]
        trg_list_CX3 = [every_qubit_dict[coord] 
                        for coord in [(1,3),(0,3),
                                      (2,4),(1,4)]
                                      ]
        trg_list_CX4 = [every_qubit_dict[coord] 
                        for coord in [(0,2),(0,3),
                                      (1,5),(1,4),
                                      (2,2),(2,3)]
                                      ]
        trg_list_CX5 = trg_list_CX4[::-1]
        trg_list_CX6 = [every_qubit_dict[coord] 
                        for coord in [(0,5),(1, 5),
                                      (1,2),(2,2)]
                                      ]
        trg_list_CX7 = [every_qubit_dict[coord] 
                        for coord in [(1,5),(0,5),
                                      (1,2),(0,2)]
                                      ]
        trg_list_CX8 = [every_qubit_dict[coord] 
                        for coord in [(2,2),(1,2)]
                                      ]
        trg_list_CX9 = [every_qubit_dict[coord] 
                        for coord in [(0,2),(1,2)]
                                      ]
        trg_list_S = [every_qubit_dict[(1,2)]]
        trg_list_CX10 = [every_qubit_dict[coord] 
                        for coord in [(0,1),(0,2),
                                      (3,2),(2,2)]
                                      ]
        trg_list_CX11 = trg_list_CX10[::-1]
        
        if pipeline:
            trg_set_R_rem = set(trg_list_R)
            trg_set_H_rem = set(trg_list_H)
            trg_set_R = trg_set_R_rem & set(trg_list_CX1)
            trg_set_H = trg_set_H_rem & set(trg_list_CX1)
            trg_set_R_rem-=trg_set_R
            trg_set_H_rem-=trg_set_H
            if self.no_hadamards:
                self.circuit.append("R",trg_set_R-trg_set_H)
                self.circuit.append("RX",trg_set_H)
            else:
                self.circuit.append("R",trg_set_R)
                self.circuit.append("H",trg_set_H)
            self.circuit.append("TICK")
            
            all_CX_trgs = [trg_list_CX1,trg_list_CX2,trg_list_CX3,trg_list_CX4,trg_list_CX5,trg_list_CX6,trg_list_CX7,trg_list_CX8,trg_list_CX9] 
            for i, trgs in enumerate(all_CX_trgs):
                self.circuit.append("CX",trgs)
                if i<len(all_CX_trgs)-1:
                    trg_set_R = trg_set_R_rem & set(all_CX_trgs[i+1])
                    trg_set_H = trg_set_H_rem & set(all_CX_trgs[i+1])
                    trg_set_R_rem-=trg_set_R
                    trg_set_H_rem-=trg_set_H
                    if self.no_hadamards:
                        self.circuit.append("R",trg_set_R-trg_set_H)
                        self.circuit.append("RX",trg_set_H)
                    else:
                        self.circuit.append("R",trg_set_R)
                        self.circuit.append("H",trg_set_H)
                self.circuit.append("TICK")


            self.circuit.append("S_DAG",trg_list_S)
            self.circuit.append("TICK")
            for trgs in [trg_list_CX9,trg_list_CX8,trg_list_CX10]:
                self.circuit.append("CX",trgs)
                if trgs == trg_list_CX8:
                    if self.no_hadamards:
                        self.circuit.append("R",trg_set_R_rem-trg_set_H_rem)
                        self.circuit.append("RX",trg_set_H_rem)
                    else:
                        self.circuit.append("R",trg_set_R_rem)
                        self.circuit.append("H",trg_set_H_rem)
                self.circuit.append("TICK")

        else:
            trg_set_R = set(trg_list_R)
            trg_set_H = set(trg_list_H)
            if self.no_hadamards:
                self.circuit.append("R",trg_set_R-trg_set_H)
                self.circuit.append("RX",trg_set_H)
            else:
                self.circuit.append("R",trg_set_R)
                self.circuit.append("H",trg_set_H)
            trg_list_R2 = [every_qubit_dict[coord] 
                            for coord in [(-1, 1),(1,1),(0,6),(3,3)]
                                        ]
            self.circuit.append("R",trg_list_R2) # this can be done parallel with the CX rounds

            self.circuit.append("TICK")
            for trgs in [trg_list_CX1,trg_list_CX2,trg_list_CX3,trg_list_CX4,trg_list_CX5,trg_list_CX6,trg_list_CX7,trg_list_CX8,trg_list_CX9]:
                self.circuit.append("CX",trgs)
                self.circuit.append("TICK")
            self.circuit.append("S_DAG",trg_list_S)
            self.circuit.append("TICK")
            for trgs in [trg_list_CX9,trg_list_CX8,trg_list_CX10]:
                self.circuit.append("CX",trgs)
                self.circuit.append("TICK")

        ## swap data qubits to their place
        trg_list_R2 = [every_qubit_dict[coord] 
                        for coord in [(-1, 1),(1,1),(0,6),(3,3)]
                                    ]
        trg_list_CX12 = [every_qubit_dict[coord] 
                        for coord in [(0,1),(-1, 1),
                                    (1,2),(1,1),
                                    (0,5),(0,6),
                                    (3,2),(3,3)]
                                    ]
        trg_list_CX13 = trg_list_CX12[::-1]
        trg_list_M = [every_qubit_dict[coord] 
                        for coord in [(0,1),
                                    (1,2),
                                    (0,5),
                                    (3,2)]
                                    ]
        self.circuit.append("CX",trg_list_CX11)
        if pipeline:
            self.circuit.append("R",trg_list_R2) # this can be done parallel with the CX rounds
        self.circuit.append("TICK")
        self.circuit.append("CX",trg_list_CX12)
        self.circuit.append("TICK")
        self.circuit.append("CX",trg_list_CX13)
        self.circuit.append("TICK")
        self.circuit.append("M",trg_list_M)

        self.meas_ind+=4
        for i in range(1,5):
            self.circuit.append("DETECTOR",
                                [stim.target_rec(-i)],
                                [])

        self.stab(d = 3, add_detectors = False)

        _, _, _, _, _, plaquettes = qubit_coord_groups(3)

        for plaquette in plaquettes:
            anc1_coord = plaquette['ancilla1_coord']
            anc1, anc2, flag = self.ancilla_dict[plaquette['ancilla1_coord']],self.ancilla_dict[plaquette['ancilla2_coord']],self.ancilla_dict[plaquette['flag_coord']]
            ## flag detectors
            self.circuit.append("DETECTOR",
                                [stim.target_rec(self.measurement_dict[("f",flag,self.time-1)]-self.meas_ind)],
                                [])
            ## Z detectors
            self.circuit.append("DETECTOR",
                                [stim.target_rec(self.measurement_dict[("Z",anc2,self.time-1)]-self.meas_ind)],
                                [0,anc1,self.time-1])
            ## X detectors
            self.circuit.append("DETECTOR",
                                [stim.target_rec(self.measurement_dict[("X",anc1,self.time-1)]-self.meas_ind)],
                                [1,anc1,self.time-1])

        # ## for the logical detectors
        self.circuit.append("MPAD",0)
        self.meas_ind+=1
        self.measurement_dict[("log_dat"+basis,0,self.time-1)] = self.meas_ind-1
        self.last_logmeas_time = self.time-1

    def stab(self, d = None, add_detectors = True):
        if not d:
            d = self.max_d
        _qubit_dict, _ancilla_dict, _, _, boundary_qubit_coords, plaquettes = qubit_coord_groups(d)

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

    def mid_circ_meas(self, d = None, basis='Z'):
        if not d:
            d = self.max_d
        _qubit_dict, _ancilla_dict, flag_qubit_coords, link_coords, boundary_qubit_coords, plaquettes = qubit_coord_groups(d)

        def find_edge_path(v0_coord,v_coord):
            edge_path = []
            vi_coord = list(v0_coord)
            while tuple(vi_coord) != v_coord:
                if vi_coord[1] > v_coord[1] and (vi_coord[0],vi_coord[1]-2) in _qubit_dict:
                    edge_path.append(self.ancilla_dict[(vi_coord[0],vi_coord[1]-1)])
                    vi_coord[1]-=2
                elif vi_coord[1] < v_coord[1] and (vi_coord[0],vi_coord[1]+2) in _qubit_dict:
                    edge_path.append(self.ancilla_dict[(vi_coord[0],vi_coord[1]+1)])
                    vi_coord[1]+=2
                elif vi_coord[0] <= v_coord[0]:
                    if (vi_coord[0]+1,vi_coord[1]+1) in _qubit_dict:
                        edge_path.append(self.ancilla_dict[(vi_coord[0],vi_coord[1]+1)])
                        vi_coord[0]+=1
                        vi_coord[1]+=1
                    elif (vi_coord[0]+1,vi_coord[1]-1) in _qubit_dict:
                        edge_path.append(self.ancilla_dict[(vi_coord[0],vi_coord[1]-1)])
                        vi_coord[0]+=1
                        vi_coord[1]-=1
                    elif (vi_coord[0]+2,vi_coord[1]) in _qubit_dict:
                        edge_path.append(self.ancilla_dict[(vi_coord[0]+1,vi_coord[1])])
                        vi_coord[0]+=2
                elif vi_coord[0] > v_coord[0]:
                    if (vi_coord[0]-1,vi_coord[1]+1) in _qubit_dict:
                        edge_path.append(self.ancilla_dict[(vi_coord[0]-1,vi_coord[1])])
                        vi_coord[0]-=1
                        vi_coord[1]+=1
                    elif (vi_coord[0]-1,vi_coord[1]-1) in _qubit_dict:
                        edge_path.append(self.ancilla_dict[(vi_coord[0]-1,vi_coord[1])])
                        vi_coord[0]-=1
                        vi_coord[1]-=1
                    elif (vi_coord[0]-2,vi_coord[1]) in _qubit_dict:
                        edge_path.append(self.ancilla_dict[(vi_coord[0]-1,vi_coord[1])])
                        vi_coord[0]-=2                    
            return edge_path


        pipeline_with_previous = self.pipeline
        
        trg_list_R1 = [self.qubit_dict[coord] for coord in boundary_qubit_coords]
        trg_list_R2 = []
        trg_list_H1 = [self.qubit_dict[coord] for coord in boundary_qubit_coords]
        trg_list_H2 =[]
        trg_list_CX1 = []
        trg_list_CX2 = []
        trg_list_CX3 = []
        trg_list_CX4 = []
        trg_list_CX5 = []
        trg_list_M1 = []
        trg_list_M2 = []
        for (q1_coord, q2_coord), a_coord, f_coord in link_coords:
            if f_coord[1]%6==2: #early shift
                if a_coord[1]%3==2: 
                    trg_list_R1+=[self.ancilla_dict[a_coord]]
                    trg_list_CX1+=[self.qubit_dict[q2_coord],self.ancilla_dict[a_coord]]
                    if f_coord in self.ancilla_dict:
                        # if not (f_coord[0]%6 == 0 and f_coord[1]%6==2) and not (f_coord[0]%6 == 3 and f_coord[1]%6==5): #not red plaquette, lower edge
                            trg_list_CX2+=[self.ancilla_dict[f_coord],self.ancilla_dict[a_coord]]
                    trg_list_CX3+=[self.qubit_dict[q1_coord],self.ancilla_dict[a_coord]]
                    trg_list_M1+=[self.qubit_dict[q2_coord]]
                    trg_list_M2+=[self.qubit_dict[q1_coord]]
                elif a_coord[1]%3==1:
                    trg_list_R2+=[self.ancilla_dict[a_coord]]
                    trg_list_CX2+=[self.qubit_dict[q1_coord],self.ancilla_dict[a_coord]]
                    if f_coord in self.ancilla_dict:
                        # if not (f_coord[0]%6 == 2 and f_coord[1]%6==2) and not (f_coord[0]%6 == 5 and f_coord[1]%6==5): #not blue plaquette, lower edge
                        # if not (f_coord[0]%6 == 2 and f_coord[1]%6==2) and not (f_coord[0]%6 == 5 and f_coord[1]%6==5) and (q1_coord in boundary_qubit_coords or q2_coord not in boundary_qubit_coords):
                            trg_list_CX4+=[self.ancilla_dict[f_coord],self.ancilla_dict[a_coord]]
                    trg_list_CX5+=[self.qubit_dict[q2_coord],self.ancilla_dict[a_coord]]
                else:
                    trg_list_R2+=[self.ancilla_dict[a_coord]]
                    if q1_coord in boundary_qubit_coords:
                        trg_list_CX2+=[self.qubit_dict[q1_coord],self.ancilla_dict[a_coord]]
                        trg_list_CX4+=[self.qubit_dict[q2_coord],self.ancilla_dict[a_coord]]
                    else:
                        trg_list_CX2+=[self.qubit_dict[q2_coord],self.ancilla_dict[a_coord]]
                        trg_list_CX4+=[self.qubit_dict[q1_coord],self.ancilla_dict[a_coord]]
                    if f_coord in self.ancilla_dict:
                        # if not (f_coord[0]%6 == 1 and f_coord[1]%6==5) and not (f_coord[0]%6 == 4 and f_coord[1]%6==2): #not green plaquette, lower edge
                            trg_list_CX3+=[self.ancilla_dict[f_coord],self.ancilla_dict[a_coord]]

                if f_coord in self.ancilla_dict and self.ancilla_dict[f_coord] not in trg_list_H2:
                    trg_list_R2+=[self.ancilla_dict[f_coord]]
                    trg_list_H2+=[self.ancilla_dict[f_coord]]

            else: #late shift
                if a_coord[1]%3==2: 
                    trg_list_R1+=[self.ancilla_dict[a_coord]]
                    trg_list_CX3+=[self.qubit_dict[q2_coord],self.ancilla_dict[a_coord]]
                    if f_coord in self.ancilla_dict:
                        # if not (f_coord[0]%6 == 0 and f_coord[1]%6==2) and not (f_coord[0]%6 == 3 and f_coord[1]%6==5): #not red plaquette, lower edge                            
                            trg_list_CX2+=[self.ancilla_dict[f_coord],self.ancilla_dict[a_coord]]    
                    trg_list_CX1+=[self.qubit_dict[q1_coord],self.ancilla_dict[a_coord]]
                    trg_list_M2+=[self.qubit_dict[q2_coord]]
                    trg_list_M1+=[self.qubit_dict[q1_coord]]
                elif a_coord[1]%3==1:
                    trg_list_R2+=[self.ancilla_dict[a_coord]]
                    trg_list_CX2+=[self.qubit_dict[q1_coord],self.ancilla_dict[a_coord]]
                    if f_coord in self.ancilla_dict:
                        # if not (f_coord[0]%6 == 2 and f_coord[1]%6==2) and not (f_coord[0]%6 == 5 and f_coord[1]%6==5): #not blue plaquette, lower edge
                            trg_list_CX3+=[self.ancilla_dict[f_coord],self.ancilla_dict[a_coord]]
                    trg_list_CX4+=[self.qubit_dict[q2_coord],self.ancilla_dict[a_coord]]
                else:
                    trg_list_R2+=[self.ancilla_dict[a_coord]]
                    if q1_coord in boundary_qubit_coords:
                        trg_list_CX2+=[self.qubit_dict[q1_coord],self.ancilla_dict[a_coord]]
                        trg_list_CX4+=[self.qubit_dict[q2_coord],self.ancilla_dict[a_coord]]
                    else:
                        trg_list_CX2+=[self.qubit_dict[q2_coord],self.ancilla_dict[a_coord]]
                        trg_list_CX5+=[self.qubit_dict[q1_coord],self.ancilla_dict[a_coord]]
                        if f_coord in self.ancilla_dict:
                            # if not (f_coord[0]%6 == 1 and f_coord[1]%6==5) and not (f_coord[0]%6 == 4 and f_coord[1]%6==2): #not green plaquette, lower edge
                                trg_list_CX4+=[self.ancilla_dict[f_coord],self.ancilla_dict[a_coord]]

                if f_coord in self.ancilla_dict and self.ancilla_dict[f_coord] not in trg_list_H2:
                    trg_list_R2+=[self.ancilla_dict[f_coord]]
                    trg_list_H2+=[self.ancilla_dict[f_coord]]

        #needed for the S/S_DAG gates
        up_vertices = list(set(self.qubit_dict[center[0],center[1]+2] 
                                for center in flag_qubit_coords 
                                if (center[0],center[1]+2) in _qubit_dict and (center[0],center[1]+2) not in boundary_qubit_coords) 
                            | set(self.qubit_dict[center[0],center[1]-4] 
                                    for center in flag_qubit_coords 
                                    if (center[0],center[1]-4) in _qubit_dict and (center[0],center[1]-4) not in boundary_qubit_coords))
        down_vertices = [self.qubit_dict[coord] for coord in _qubit_dict.keys() if (self.qubit_dict[coord] not in up_vertices) and (coord not in boundary_qubit_coords)]
        if not self.measure_Clifford_on_data:
            if basis == 'Z':
                self.circuit.append("H",[self.qubit_dict[coord] for coord in _qubit_dict if coord not in boundary_qubit_coords])
            elif basis == 'Y':
                self.circuit.append("S_DAG",up_vertices)
                self.circuit.append("S",down_vertices)

        # if not pipeline_with_previous:
        #     self.circuit.append("TICK")

        if self.no_hadamards:
            self.circuit.append("R",set(trg_list_R1)-set(trg_list_H1))
            self.circuit.append("RX",trg_list_H1)
        else:
            self.circuit.append("R",trg_list_R1)
            self.circuit.append("H",trg_list_H1)
        if pipeline_with_previous:
            self.circuit.append("TICK")

        if self.no_hadamards:
            self.circuit.append("R",set(trg_list_R2)-set(trg_list_H2))
            self.circuit.append("RX",trg_list_H2)
        else:
            self.circuit.append("R",trg_list_R2)
            self.circuit.append("H",trg_list_H2)
        if not pipeline_with_previous:
            self.circuit.append("TICK")
        self.circuit.append("CX",trg_list_CX1)
        self.circuit.append("TICK")

        for trgs in [trg_list_CX2,trg_list_CX3,trg_list_CX4]:
            if trgs:
                self.circuit.append("CX",trgs)
                self.circuit.append("TICK")

        self.circuit.append("CX",trg_list_CX5)
        if not pipeline_with_previous:
            self.circuit.append("TICK")




        if self.measure_Clifford_on_data:
            if basis == 'Z':
                self.circuit.append("H",[self.qubit_dict[coord] for coord in _qubit_dict if coord not in boundary_qubit_coords])
            elif basis == 'Y':
                self.circuit.append("S_DAG",up_vertices)
                self.circuit.append("S",down_vertices)





        if self.no_hadamards:
            self.circuit.append("MX",trg_list_M1)
        else:
            self.circuit.append("H",trg_list_M1)
            self.circuit.append("M",trg_list_M1)
            self.circuit.append("H",trg_list_M1)
        if pipeline_with_previous:
            self.circuit.append("TICK")


        for q in trg_list_M1:
            self.measurement_dict[("log_dat"+basis,q,self.time)] = self.meas_ind
            self.meas_ind+=1
        if self.no_hadamards:
            self.circuit.append("MX",trg_list_M2)
        else:
            self.circuit.append("H",trg_list_M2)
            self.circuit.append("M",trg_list_M2)
            self.circuit.append("H",trg_list_M2)

        if self.measure_Clifford_on_data:
            if basis == 'Z':
                self.circuit.append("H",[self.qubit_dict[coord] for coord in _qubit_dict if coord not in boundary_qubit_coords])
            elif basis == 'Y':
                self.circuit.append("S",up_vertices)
                self.circuit.append("S_DAG",down_vertices)

        if not pipeline_with_previous:
            self.circuit.append("TICK")

        self.circuit.append("CX",trg_list_CX1)
        self.circuit.append("TICK")

        for q in trg_list_M2:
            self.measurement_dict[("log_dat"+basis,q,self.time)] = self.meas_ind
            self.meas_ind+=1


        logical_support = set(_qubit_dict.keys())-set(boundary_qubit_coords)
        ancilla_coords_in_logical = [plaquette['ancilla1_coord'] for plaquette in plaquettes if len(logical_support & set(plaquette['qubits1_coord']))%2]
        
        if ("log_dat"+basis,self.qubit_dict[(-1,1)],self.last_logmeas_time) in self.measurement_dict:
            self.circuit.append("DETECTOR",
                                    [stim.target_rec(
                                        self.measurement_dict[("log_dat"+basis,self.qubit_dict[coord],self.time)] - self.meas_ind) 
                                        for coord in _qubit_dict.keys()]
                                    +[stim.target_rec(
                                        self.measurement_dict[("log_dat"+basis,self.qubit_dict[coord],self.last_logmeas_time)] - self.meas_ind) 
                                        for coord in _qubit_dict.keys() if ("log_dat"+basis,self.qubit_dict[coord],self.last_logmeas_time) in self.measurement_dict]
                                    + (basis!="Z")*[stim.target_rec(
                                        self.measurement_dict[("X",self.ancilla_dict[coord],t)] - self.meas_ind) 
                                        for t in range(self.last_logmeas_time,self.time) for coord in ancilla_coords_in_logical],
                                    [-10,-10,-10]
                                    )
        else:
            self.circuit.append("OBSERVABLE_INCLUDE",
                                    [stim.target_rec(
                                        self.measurement_dict[("log_dat"+basis,self.qubit_dict[coord],self.time)] - self.meas_ind) 
                                        for coord in _qubit_dict.keys()]
                                    ,0)


        self.last_logmeas_time = self.time

        for trgs in [trg_list_CX2,trg_list_CX3,trg_list_CX4]:
            if trgs:
                self.circuit.append("CX",trgs)
            if trgs==trg_list_CX4 and pipeline_with_previous:
                for coord in boundary_qubit_coords:
                    if coord[0]>1:
                        if self.no_hadamards:
                            self.circuit.append("MX",self.qubit_dict[coord])
                        else:
                            self.circuit.append("H",self.qubit_dict[coord])
                            self.circuit.append("M",self.qubit_dict[coord])
                        trg_list_H1 = list(set(trg_list_H1)-set([self.qubit_dict[coord]]))
                        self.measurement_dict[("X_data",self.qubit_dict[coord],self.time)] = self.meas_ind
                        self.circuit.append("DETECTOR",stim.target_rec(-1), [])
                        self.meas_ind += 1
            self.circuit.append("TICK")

        self.circuit.append("CX",trg_list_CX5)
        anc_to_meas = [self.ancilla_dict[coord] for coord in _ancilla_dict]
        trg_set_M1 = set(anc_to_meas)-set(trg_list_CX5)
        trg_set_M2 = set(anc_to_meas)-trg_set_M1

        if not pipeline_with_previous:
            self.circuit.append("TICK")
            for coord in boundary_qubit_coords:
                if coord[0]>1:
                    if self.no_hadamards:
                        self.circuit.append("MX",self.qubit_dict[coord])
                    else:
                        self.circuit.append("H",self.qubit_dict[coord])
                        self.circuit.append("M",self.qubit_dict[coord])
                    trg_list_H1 = list(set(trg_list_H1)-set([self.qubit_dict[coord]]))
                    self.measurement_dict[("X_data",self.qubit_dict[coord],self.time)] = self.meas_ind
                    self.circuit.append("DETECTOR",stim.target_rec(-1), [])
                    self.meas_ind += 1

        if self.no_hadamards:
            self.circuit.append("MX",trg_list_H1+trg_list_H2)
            for i,a in enumerate(trg_list_H1+trg_list_H2):
                if a in [self.qubit_dict[coord] for coord in boundary_qubit_coords]:
                    self.measurement_dict[("X_data",a,self.time)] = self.meas_ind
                    self.circuit.append("DETECTOR",stim.target_rec(i-len(trg_list_H1+trg_list_H2)), [])
                else:
                    self.measurement_dict[("log_anc",a,self.time)] = self.meas_ind
                self.meas_ind+=1
            self.circuit.append("M",trg_set_M1-set(trg_list_H1+trg_list_H2))
            for a in trg_set_M1-set(trg_list_H1+trg_list_H2):
                self.measurement_dict[("log_anc",a,self.time)] = self.meas_ind
                self.meas_ind+=1
        else:
            self.circuit.append("H",trg_list_H1+trg_list_H2)
            self.circuit.append("M",trg_set_M1)
            for a in trg_set_M1:
                self.measurement_dict[("log_anc",a,self.time)] = self.meas_ind
                self.meas_ind+=1
            for coord in boundary_qubit_coords:
                if self.qubit_dict[coord] in trg_list_H1:
                    self.circuit.append("M",self.qubit_dict[coord])
                    self.measurement_dict[("X_data",self.qubit_dict[coord],self.time)] = self.meas_ind
                    self.circuit.append("DETECTOR",stim.target_rec(-1), [])
                    self.meas_ind += 1
        if pipeline_with_previous:
            self.circuit.append("TICK")

        self.circuit.append("M",trg_set_M2)
        for a in trg_set_M2:
            self.measurement_dict[("log_anc",a,self.time)] = self.meas_ind
            self.meas_ind+=1

        if not pipeline_with_previous:
            self.circuit.append("TICK")

        #cycle checks
        for center in flag_qubit_coords:
            anc_list = []
            for rel_coord in [[-1,2],[-1,-2],[1,0],[-1,0],[0,1],[0,-1]]:
                anc_coord = (center[0] + rel_coord[0], center[1] + rel_coord[1])
                if anc_coord in _ancilla_dict:
                    anc_list+=[self.ancilla_dict[anc_coord]]
            self.circuit.append("DETECTOR",
                                [stim.target_rec(
                                    self.measurement_dict[("log_anc",anc,self.time)] - self.meas_ind) 
                                    for anc in anc_list], [])
        #flags
        for center in flag_qubit_coords:
            self.circuit.append("DETECTOR",stim.target_rec(self.measurement_dict[("log_anc",self.ancilla_dict[center],self.time)] - self.meas_ind), [])


        ##feed-forwards to fix up the subspace
        v0_coord = (int((d-1)/2),int(3*(d-1)/2))
        ffX_rounds = [[]]
        for coord in _qubit_dict.keys():
            if coord not in boundary_qubit_coords:
                ff_acnillas = find_edge_path(v0_coord,coord)
                for ff_a in ff_acnillas:
                    trg = [stim.target_rec(self.measurement_dict[("log_anc",ff_a,self.time)]-self.meas_ind),self.qubit_dict[coord]]
                    FF_scheduled = False
                    ff_round_idx = 0
                    while not FF_scheduled:
                        if trg[1] not in ffX_rounds[ff_round_idx]:
                            ffX_rounds[ff_round_idx]+= trg
                            FF_scheduled = True
                        else:
                            ff_round_idx+=1
                            if len(ffX_rounds)==ff_round_idx:
                                ffX_rounds+=[[]]

        if not self.measure_Clifford_on_data:
            for trgs in ffX_rounds:
                self.circuit.append("CX",trgs)
            ##there is enough information from the previous measurement round to pipeline this feed forward

        if self.measure_Clifford_on_data:
            for trgs in ffX_rounds:
                self.circuit.append("C"+basis,trgs)

        if not self.measure_Clifford_on_data:
            if basis == 'Z':
                self.circuit.append("H",[self.qubit_dict[coord] for coord in _qubit_dict if coord not in boundary_qubit_coords])
            elif basis == 'Y':
                self.circuit.append("S",up_vertices)
                self.circuit.append("S_DAG",down_vertices)

    def noiseless_final_meas(self, d = None, basis='Y'):
        if not d:
            d = self.max_d
        _qubit_dict, _, _, _, boundary_qubit_coords, _ = qubit_coord_groups(d)
        
        qubits = [self.qubit_dict[coord] for coord in _qubit_dict.keys() if coord not in boundary_qubit_coords]
        pauli_string = basis+str(qubits[0])
        for q in qubits[1:]:
            pauli_string+="*"+basis+str(q)
        self.circuit.append("MPP",[stim.PauliString(pauli_string)])
        self.measurement_dict[("final_log_meas",self.time)] = self.meas_ind
        self.meas_ind+=1
        self.circuit.append("OBSERVABLE_INCLUDE",stim.target_rec(-1),0)

    def grow(self, d1, d2):
        _qubit_dict0, _ancilla_dict0, flag_qubit_coords0, _, boundary_qubit_coords0, _ = qubit_coord_groups(d1)
        _qubit_dict, _, flag_qubit_coords, _, boundary_qubit_coords, plaquettes = qubit_coord_groups(d2)

        pipeline_with_previous = self.pipeline

        red_centers = []
        for center in set(flag_qubit_coords)-set(flag_qubit_coords0):
            if center[0]%6 == 0 and center[1]%6==2:
                red_centers+=[center]
            if center[0]%6 == 3 and center[1]%6==5:
                red_centers+=[center]

        green_centers = []
        for center in set(flag_qubit_coords)-set(flag_qubit_coords0):
            if center[0]%6 == 1 and center[1]%6==5:
                green_centers+=[center]
            if center[0]%6 == 4 and center[1]%6==2:
                green_centers+=[center]

        ### Bell prep with unitary circuit
        ## Bell link on the tip with poor connectivity
        tip_coord = [coord for coord in boundary_qubit_coords if coord not in boundary_qubit_coords0 and coord[1]<coord[0]-1][0]
        q1,q2,a = self.qubit_dict[tip_coord[0],tip_coord[1]+2], self.qubit_dict[tip_coord[0]-2,tip_coord[1]],self.ancilla_dict[tip_coord[0]-1,tip_coord[1]+1]
        a1,a2 = self.ancilla_dict[tip_coord[0],tip_coord[1]+1], self.ancilla_dict[tip_coord[0]-1,tip_coord[1]]
        #preparing a Bell state on the NNN ancillas
        trg_list_R1 = [q1,q2,a1,a2,a]
        trg_list_H1 = [a1]
        trg_list_CX1 = [a1,a]
        trg_list_CX2 = [a,a2]
        trg_list_CX3 = [a1,a]
        #swapping to the qubits
        trg_list_CX4 = [a1,q1,a2,q2]
        trg_list_CX5 = [q1,a1,q2,a2]
        trg_list_CX6 = []     
        Bell_links = set((((center[0]+1,center[1]-1),
                           (center[0],center[1]-2)),
                           (center[0],center[1]-1)) for center in green_centers)
        Bell_links ^= set((((center[0]-1,center[1]-1),
                            (center[0]-1,center[1]+1)),
                            (center[0]-1,center[1])) for center in green_centers)
        #adding links to the boundary of truncated plaquettes 
        Bell_links ^= set((((center[0]+1,center[1]+1),
                            (center[0],center[1]+2)),
                            (center[0]+1,center[1]+2)) for center in flag_qubit_coords0 if (center[0]+1,center[1]+1) in boundary_qubit_coords0)
        trg_list_R2 = []
        trg_list_R3 = []
        trg_list_H2 = []
        for (q1_coord,q2_coord),a_coord in Bell_links:
            q1,q2,a = self.qubit_dict[q1_coord], self.qubit_dict[q2_coord], self.ancilla_dict[a_coord]
            if q1_coord in boundary_qubit_coords0:
                trg_list_R2+=[q2,a]
                trg_list_R3+=[q1]
                trg_list_H2+=[q2]
                trg_list_CX3+=[q2,a]
                trg_list_CX5+=[a,q1]
                trg_list_CX6+=[q2,a]
            else:
                trg_list_R2+=[q1,q2,a]
                trg_list_H2+=[q2]
                trg_list_CX3+=[q2,a]
                trg_list_CX4+=[a,q1]
                trg_list_CX5+=[q2,a]

        if pipeline_with_previous:
            self.circuit.append("TICK") # the last tick of the previous logical check round is removed
            tick_inds = []
            for i,inst in enumerate(self.circuit[::-1],start = 1):
                if inst.name=="TICK":
                    tick_inds+=[i]
            pipelined_circ = self.circuit[:-tick_inds[5]]
            if self.no_hadamards:
                pipelined_circ.append("R",set(trg_list_R1)-set(trg_list_H1))
                pipelined_circ.append("RX",trg_list_H1)
            else:
                pipelined_circ.append("R",trg_list_R1)
                pipelined_circ.append("H",trg_list_H1)
            for inst in self.circuit[-tick_inds[5]:-tick_inds[4]]:
                pipelined_circ.append(inst)
            pipelined_circ.append("CX",trg_list_CX1)
            for inst in self.circuit[-tick_inds[4]:-tick_inds[3]]:
                pipelined_circ.append(inst)
            pipelined_circ.append("CX",trg_list_CX2)
            if self.no_hadamards:
                pipelined_circ.append("R",set(trg_list_R2)-set(trg_list_H2))
                pipelined_circ.append("RX",trg_list_H2)
            else:
                pipelined_circ.append("R",trg_list_R2)
                pipelined_circ.append("H",trg_list_H2)
            for inst in self.circuit[-tick_inds[3]:-tick_inds[2]]:
                pipelined_circ.append(inst)
            pipelined_circ.append("CX",trg_list_CX3)
            for inst in self.circuit[-tick_inds[2]:-tick_inds[1]]:
                pipelined_circ.append(inst)
            pipelined_circ.append("R",trg_list_R3)
            pipelined_circ.append("CX",trg_list_CX4)
            pipelined_circ.append("TICK")
            pipelined_circ.append("CX",trg_list_CX5)
            for inst in self.circuit[-tick_inds[1]+1:-tick_inds[0]]:
                pipelined_circ.append(inst)
            for inst in self.circuit[-tick_inds[0]:]:
                pipelined_circ.append(inst)
            self.circuit = pipelined_circ
            self.circuit.append("CX",trg_list_CX6)

        else:
            if self.no_hadamards:
                trg_list_R = trg_list_R1+trg_list_R2+trg_list_R3
                trg_list_RX = trg_list_H1+trg_list_H2
                self.circuit.append("R",trg_list_R)
                self.circuit.append("RX",trg_list_RX)            
            else:
                trg_list_R = trg_list_R1+trg_list_R2+trg_list_R3
                trg_list_H = trg_list_H1+trg_list_H2
                self.circuit.append("R",trg_list_R)
                self.circuit.append("H",trg_list_H)
            for trg_list in [trg_list_CX1,trg_list_CX2,trg_list_CX3,trg_list_CX4,trg_list_CX5,trg_list_CX6]:
                self.circuit.append("TICK")
                self.circuit.append("CX",trg_list)

        self.stab(d=d2,add_detectors=False)

        for plaquette in plaquettes:
            anc1_coord = plaquette['ancilla1_coord']
            anc1, anc2, flag = self.ancilla_dict[plaquette['ancilla1_coord']],self.ancilla_dict[plaquette['ancilla2_coord']],self.ancilla_dict[plaquette['flag_coord']]
            ## flag detectors
            self.circuit.append("DETECTOR",
                                [stim.target_rec(self.measurement_dict[("f",flag,self.time-1)]-self.meas_ind)],
                                [])

            ## Z detectors
            if plaquette['flag_coord'] not in red_centers:
                self.circuit.append("DETECTOR",
                                    [stim.target_rec(self.measurement_dict[("Z",anc2,self.time-1)]-self.meas_ind)],
                                    [0,anc1,self.time-1])

            ## Z detector fixup on the red plaquettes
            if plaquette['flag_coord'] in red_centers:
                q1, q2 = self.qubit_dict[plaquette['flag_coord'][0],plaquette['flag_coord'][1]+2], self.qubit_dict[plaquette['flag_coord'][0],plaquette['flag_coord'][1]+4]
                self.circuit.append("CX",[stim.target_rec(self.measurement_dict[("Z",anc2,self.time-1)]-self.meas_ind),q1])
                self.circuit.append("CX",[stim.target_rec(self.measurement_dict[("Z",anc2,self.time-1)]-self.meas_ind),q2])
                del self.measurement_dict[("Z",anc2,self.time-1)] # not to confuse the next stabilizer measurement

            ## d1-code X detectors
            if plaquette['flag_coord'] in flag_qubit_coords0:
                # stabilizers along the base
                if anc1_coord[0]==0:
                    self.circuit.append("DETECTOR",
                                        [stim.target_rec(self.measurement_dict[("X",anc1,self.time-1)]-self.meas_ind),
                                        stim.target_rec(self.measurement_dict[("X",anc1,self.time-2)]-self.meas_ind),
                                        stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(anc1_coord[0]+2,anc1_coord[1])],self.time-2)]-self.meas_ind)],
                                        [1,anc1,self.time-1])
                    
                # stabilizers along the legs
                elif (anc1_coord[0],anc1_coord[1]-1) not in _qubit_dict0 or (anc1_coord[0],anc1_coord[1]+3) not in _qubit_dict0: 
                    # the flag above exists (the ancilla could exist due to the link)
                    if (anc1_coord[0]-2,anc1_coord[1]+1) in _ancilla_dict0: 
                        self.circuit.append("DETECTOR",
                                            [stim.target_rec(self.measurement_dict[("X",anc1,self.time-1)]-self.meas_ind),
                                            stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(anc1_coord[0]-2,anc1_coord[1])],self.time-2)]-self.meas_ind)],
                                            [1,anc1,self.time-1])
                    else:
                        self.circuit.append("DETECTOR",
                                            [stim.target_rec(self.measurement_dict[("X",anc1,self.time-1)]-self.meas_ind)],
                                            [1,anc1,self.time-1])
                # stabilizers in the bulk
                else:
                    if (anc1_coord[0]-2,anc1_coord[1]+1) in _ancilla_dict0: 
                        self.circuit.append("DETECTOR",
                                            [stim.target_rec(self.measurement_dict[("X",anc1,self.time-1)]-self.meas_ind),
                                            stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(anc1_coord[0]-2,anc1_coord[1])],self.time-2)]-self.meas_ind),
                                            stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(anc1_coord[0]+2,anc1_coord[1])],self.time-2)]-self.meas_ind)
                                            ],
                                            [1,anc1,self.time-1])
                    elif (anc1_coord[0]+2,anc1_coord[1]) in _ancilla_dict0:
                        self.circuit.append("DETECTOR",
                                            [stim.target_rec(self.measurement_dict[("X",anc1,self.time-1)]-self.meas_ind),
                                            stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(anc1_coord[0]+2,anc1_coord[1])],self.time-2)]-self.meas_ind)],
                                            [1,anc1,self.time-1])

            ## X detectors at the boundary
            if plaquette['flag_coord'] in flag_qubit_coords0:
                # stabilizers along the upper leg
                if (anc1_coord[0],anc1_coord[1]+3) not in _qubit_dict0: 
                    # the flag above exists (the ancilla could exist due to the link)
                    if (anc1_coord[0]-2,anc1_coord[1]+1) in _ancilla_dict0: 
                        self.circuit.append("DETECTOR",
                                            [stim.target_rec(self.measurement_dict[("X",anc1,self.time-1)]-self.meas_ind),
                                            stim.target_rec(self.measurement_dict[("X",self.ancilla_dict[(anc1_coord[0]-2,anc1_coord[1])],self.time-2)]-self.meas_ind),
                                            ],
                                            [1,anc1,self.time-1])
                    else:
                        self.circuit.append("DETECTOR",
                                            [stim.target_rec(self.measurement_dict[("X",anc1,self.time-1)]-self.meas_ind),
                                             ],
                                            [1,anc1,self.time-1])

            ## X detectors of the grown part
            if plaquette['flag_coord'] in (set(flag_qubit_coords)-set(flag_qubit_coords0)) - set(red_centers): #green and blue plaquettes
                center = plaquette['flag_coord']
                self.circuit.append("DETECTOR",
                                    [stim.target_rec(self.measurement_dict[("X",anc1,self.time-1)]-self.meas_ind)],
                                    [1,anc1,self.time])

        logical_support = set(_qubit_dict.keys())-set(boundary_qubit_coords)
        ancilla_coords_in_logical = [plaquette['ancilla1_coord'] for plaquette in plaquettes if len(logical_support & set(plaquette['qubits1_coord']))%2]
        if (any([("log_datX",self.qubit_dict[(-1,1)],t) in self.measurement_dict for t in range(self.time+1)])
            or any([("log_datY",self.qubit_dict[(-1,1)],t) in self.measurement_dict for t in range(self.time+1)])):
            self.circuit.append("OBSERVABLE_INCLUDE",
                                    [stim.target_rec(
                                        self.measurement_dict[("X",self.ancilla_dict[coord],self.time-1)] - self.meas_ind) 
                                        for coord in ancilla_coords_in_logical]
                                    ,0)

def get_stim_circ_w_growing(d = 3, basis = "Y", extra_ticks=False, pipeline = True, measure_Clifford_on_data = False):
    d_time = d
    d_spatial = d+1-d%2
    cc = ColorCode(d_spatial)
    cc.pipeline = pipeline
    cc.measure_Clifford_on_data = measure_Clifford_on_data
    cc.unitary_prep(3, basis=basis)
    if extra_ticks:
        cc.circuit.append("TICK")
        cc.circuit.append("TICK")
    cc.mid_circ_meas(3, basis=basis)

    for dd in range(3,d_spatial,2):
        if extra_ticks:
            cc.circuit.append("TICK")
        cc.grow(dd,dd+2)
        if extra_ticks:
            cc.circuit.append("TICK")
        cc.mid_circ_meas(dd+2,basis=basis)

    for _ in range(d_time-d_time//2-1):
        if extra_ticks:
            cc.circuit.append("TICK")
        cc.stab(d_spatial)
        if extra_ticks:
            cc.circuit.append("TICK")
        cc.mid_circ_meas(d_spatial,basis=basis)

    ##separates the noiseless measurements
    cc.circuit.append("TICK")
    cc.circuit.append("TICK")
    cc.stab(d_spatial)
    cc.noiseless_final_meas(d_spatial,basis=basis)
    return cc.circuit