import cirq
import numpy as np
import networkx as nx

def Trotter_Step_Circuit(Number_of_Fock_States, Number_of_Bosonic_Modes, Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent):
    qubits = cirq.LineQubit.range(Number_of_Bosonic_Modes * (Number_of_Fock_States+1))
    circuit = cirq.Circuit()

    for j in range(Number_of_Bosonic_Modes):
        for i in range(0, Number_of_Fock_States-1, 2):
            circuit.append(cirq.ISwapPowGate(exponent=-2*Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

        for i in range(1, Number_of_Fock_States-1, 2):
            circuit.append(cirq.ISwapPowGate(exponent=-2*Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

    for j in range(0, Number_of_Bosonic_Modes-1, 2):
        circuit.append(cirq.ISwapPowGate(exponent=-4*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range(1, Number_of_Bosonic_Modes-1, 2):
            circuit.append(cirq.ISwapPowGate(exponent=-4*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range (Number_of_Bosonic_Modes):
         circuit.append(cirq.rx(2*Time*Spin_Boson_Interaction_Coefficent).controlled()(qubits[(j+1)*Number_of_Fock_States-1],qubits[Number_of_Fock_States*Number_of_Bosonic_Modes+j]))

    return circuit, qubits

def Full_Circuit(Trotter_circuit, qubits, Number_of_Bosonic_Modes, Number_of_Fock_States, Number_of_Trotter_Steps):
    circuit = cirq.Circuit()
    for j in range(Number_of_Bosonic_Modes):
        circuit.append(cirq.X(qubits[j*Number_of_Fock_States]))
    circuit.append([Trotter_circuit]*Number_of_Trotter_Steps)
    circuit.append(cirq.measure(*qubits, key='m'))

    return circuit

_STRATIFY_CATEGORIES = [lambda op: len(op.qubits) == 2, lambda op: len(op.qubits) == 1]

def Compile_For_Willow(circuit, qubits, Qubit_Mapping):
    Qubit_Map = dict(zip(qubits, Qubit_Mapping))
    circuit = circuit.transform_qubits(Qubit_Map)
    circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    circuit = cirq.map_operations(circuit, lambda op, _: cirq.CZ(*op.qubits) if isinstance(op.gate, cirq.CZPowGate) and abs(op.gate.exponent) == 1 else op)  # CZ^-1 == CZ
    circuit = cirq.merge_single_qubit_gates_to_phxz(circuit)   # merges per-qubit runs, not per-moment
    circuit = cirq.drop_negligible_operations(circuit)
    circuit = cirq.stratified_circuit(circuit, categories=_STRATIFY_CATEGORIES)
    return cirq.drop_empty_moments(circuit)

def Sample_On_Quantum_Engine(circuit, engine, processor_id, Number_of_Shots):

    processor = engine.get_processor(processor_id)
    sampler = processor.get_sampler()
    print("Sampling circuit on Quantum Engine...\n")
    results = sampler.run(circuit, repetitions=Number_of_Shots)

    return results


def UnaryPostSelection(results, Number_of_Fock_States, Number_of_Bosonic_Modes):
    Boson_Results = results[:, :Number_of_Bosonic_Modes*Number_of_Fock_States].reshape(-1, Number_of_Bosonic_Modes, Number_of_Fock_States)     # (shots, modes, Fock states)
    Unary_Mask = np.all(Boson_Results.sum(axis=2) == 1, axis=1)          # True if every register has exactly one qubit in |1>
    results = results[Unary_Mask]
    return results