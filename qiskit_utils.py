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
import qiskit, qiskit_aer
from color_code import ColorCode
from utils import qubit_coord_groups

def get_stim_circ_d3_nofinalmeas(basis = "Y"):
    cc = ColorCode(3)
    cc.unitary_prep(3, basis=basis)
    cc.mid_circ_meas(3, basis=basis)
    cc.stab(3)
    cc.mid_circ_meas(3, basis=basis)
    cc.circuit.append('TICK')
    cc.circuit.append('TICK')
    return cc.circuit

def stim_to_qiskit(stim_circ,replace_S_with_T=True):
    # this is based on https://github.com/qiskit-community/qiskit-qec/blob/main/src/qiskit_qec/circuits/stim_code_circuit.py#L47
    qc = qiskit.QuantumCircuit(stim_circ.num_qubits,stim_circ.num_measurements)

    meas_ind = 0
    dets=[]
    log=[]
    sv_saved = False

    for ind,inst in enumerate(stim_circ):
        qubits = [trg.qubit_value for trg in inst.targets_copy()]
        meas_targets = [meas_ind+trg.value for trg in inst.targets_copy()]
        if inst.name in {'DEPOLARIZE2','MPP','QUBIT_COORDS','X_ERROR'}:
            #ignore
            qc = qc
        elif inst.name=='DETECTOR':
            dets.append(meas_targets)
        elif inst.name=='OBSERVABLE_INCLUDE':
            if not sv_saved:
                log+=[meas_ind+trg.value for trg in inst.targets_copy()]
        elif inst.name=='TICK':
            if stim_circ[ind-1].name=='TICK':
                sv_saved = True
                qc.save_statevector()
            qc.barrier()
        elif inst.name=='DEPOLARIZE1':
            qc.id(qubits)
        elif inst.name=='M':
            qc.measure(qubits,range(meas_ind,meas_ind+len(qubits)))
            meas_ind+=len(qubits)
        elif inst.name=='MPAD':
            assert inst.targets_copy()[0].qubit_value == 0
            meas_ind+=1
        elif inst.name=='R':
            qc.reset(qubits)
        elif inst.name=='S':
            if replace_S_with_T:
                qc.t(qubits)
            else:
                qc.s(qubits)
        elif inst.name=='S_DAG':
            if replace_S_with_T:
                qc.tdg(qubits)
            else:
                qc.sdg(qubits)
        elif inst.name=='H':
            if qubits:
                qc.h(qubits)
        elif inst.name=='CX':
            if None not in qubits:
                qc.cx([qubits[i] for i in range(0,len(qubits),2)],[qubits[i] for i in range(1,len(qubits),2)])
            else:
                for i in range(0,len(qubits),2):
                    with qc.if_test((meas_ind+inst.targets_copy()[i].value,1)) as else_:
                        qc.x(qubits[i+1])
        else:
            NotImplementedError()

    return qc,dets,log



def Tstate_fidelity_sv(qc, detectors, log_obs, num_shots = 100, prob_1 = 0., prob_2 = 0., prob_ro = 0., replace_S_with_T = True):
    X_op_circ = qiskit.QuantumCircuit(10)
    X_op_circ.x([0, 2, 3, 4, 5, 6, 9])
    log_x = qiskit.quantum_info.Operator(X_op_circ)
    Y_op_circ = qiskit.QuantumCircuit(10)
    Y_op_circ.y([0, 2, 3, 4, 5, 6, 9])
    log_y = qiskit.quantum_info.Operator(Y_op_circ)
    if replace_S_with_T:
        log_T = (log_x+log_y)/np.sqrt(2)
    else:
        log_T = log_y

    qubit_dict,_,_,_,_,plaquettes = qubit_coord_groups(3)

    supports = [[qubit_dict[coord] for coord in p['qubits1_coord']+p['qubits2_coord'] if coord] for p in plaquettes]

    stab_ops = []
    #X detectors are added with the final measurement
    for sup in supports:
        circ = qiskit.QuantumCircuit(10)
        circ.x(sup)
        stab_ops.append(qiskit.quantum_info.Operator(circ))
    for sup in supports:
        circ = qiskit.QuantumCircuit(10)
        circ.z(sup)
        stab_ops.append(qiskit.quantum_info.Operator(circ))

    def check_state_vector(state_vec,err):
        # get rid of ancillas
        noanc_densitymat = qiskit.quantum_info.partial_trace(state_vec,range(10,25))

        # check stabilizers
        stab_evs = []
        for stab in stab_ops:
            stab_evs.append(np.round(noanc_densitymat.expectation_value(stab)).real)

        # calculate log outcome if stabilizers pass 
        if all(np.array(stab_evs)>1-0.999*err):
            log_out = (noanc_densitymat.expectation_value(log_T)).real
        else:
            log_out = 1j

        return log_out

    statevector_simulator = qiskit_aer.AerSimulator(method="statevector")


    # Depolarizing quantum errors
    error_1 = qiskit_aer.noise.depolarizing_error(4/3*prob_1, 1)
    error_2 = qiskit_aer.noise.depolarizing_error(16/15*prob_2, 2)
    error_ro = qiskit_aer.noise.ReadoutError([[1-prob_ro,prob_ro],[prob_ro,1-prob_ro]])
    error_res = qiskit_aer.noise.reset_error(1-prob_ro,prob_ro)

    # Add errors to noise model
    noise_model = qiskit_aer.noise.NoiseModel()
    # noise_model.add_all_qubit_quantum_error(error_1, ['s','sdg','t','tdg','id'])
    noise_model.add_all_qubit_quantum_error(error_1, ['id']) #for every DEP1 channel we add an identity gate in qiskit, so we don't need additional errors on single qubit gates
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    noise_model.add_all_qubit_quantum_error(error_res,['reset'])
    noise_model.add_all_qubit_readout_error(error_ro)
    tcirc = qiskit.transpile(qc, statevector_simulator, basis_gates=['id','measure','reset','s','sdg','t','tdg','x','h','cx','save_statevector','if_else'], optimization_level=0)
    ps_yield = 0
    ps_shots = 0
    T_ev_sum = 0
    for k in range(num_shots):
        # result = statevector_simulator.run(tcirc,shots = 1, noise_model = noise_model, device = "GPU").result()
        result = statevector_simulator.run(tcirc,shots = 1, noise_model = noise_model).result()
        sample = list(result.get_counts().keys())[0]

        if all([(sum([int(sample[-meas-1]) for meas in det])%2)==0 for det in detectors]):
            sv = result.get_statevector()
            ps_yield += 1
            log_out = check_state_vector(state_vec=sv, err=min(prob_1/3+(prob_1==0),prob_2/15+(prob_2==0),prob_ro+(prob_ro==0))) #log_out is -1,0, or 1
            if log_out.imag==0:
                ps_shots += 1
                midcirc_logout = (sum([int(sample[-meas-1]) for meas in log_obs])%2)
                T_ev_sum += (-1)**midcirc_logout*log_out

    fail_sum = (ps_shots-T_ev_sum)/2 #failed shots are the ones when T_ev<1
    return {'ps_shots':[ps_shots,ps_yield],
            'fail_sum': fail_sum}
