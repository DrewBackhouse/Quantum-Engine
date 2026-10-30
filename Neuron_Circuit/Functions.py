import cirq
import cirq_google
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math

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

def Trotter_Step_Circuit_2nd_Order(Number_of_Fock_States, Number_of_Bosonic_Modes, Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent):
    qubits = cirq.LineQubit.range(Number_of_Bosonic_Modes * (Number_of_Fock_States+1))
    circuit = cirq.Circuit()

    for j in range(Number_of_Bosonic_Modes):
        for i in range(0, Number_of_Fock_States-1, 2):
            circuit.append(cirq.ISwapPowGate(exponent=-Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

    for j in range(0, Number_of_Bosonic_Modes-1, 2):
        circuit.append(cirq.ISwapPowGate(exponent=-2*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range(Number_of_Bosonic_Modes):
        for i in range(1, Number_of_Fock_States-1, 2):
            circuit.append(cirq.ISwapPowGate(exponent=-Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

    for j in range(1, Number_of_Bosonic_Modes-1, 2):
        circuit.append(cirq.ISwapPowGate(exponent=-2*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range (Number_of_Bosonic_Modes):
         circuit.append(cirq.rx(2*Time*Spin_Boson_Interaction_Coefficent).controlled()(qubits[(j+1)*Number_of_Fock_States-1],qubits[Number_of_Fock_States*Number_of_Bosonic_Modes+j]))

    for j in range(Number_of_Bosonic_Modes):
        for i in range(1, Number_of_Fock_States-1, 2):
            circuit.append(cirq.ISwapPowGate(exponent=-Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))
    
    for j in range(1, Number_of_Bosonic_Modes-1, 2):
        circuit.append(cirq.ISwapPowGate(exponent=-2*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range(Number_of_Bosonic_Modes):
        for i in range(0, Number_of_Fock_States-1, 2):
            circuit.append(cirq.ISwapPowGate(exponent=-Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))
    
    for j in range(0, Number_of_Bosonic_Modes-1, 2):
        circuit.append(cirq.ISwapPowGate(exponent=-2*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    return circuit, qubits

def Full_Circuit(Trotter_circuit, qubits, Number_of_Bosonic_Modes, Number_of_Fock_States, Number_of_Trotter_Steps):
    circuit = cirq.Circuit()
    for j in range(Number_of_Bosonic_Modes):
        circuit.append(cirq.X(qubits[j*Number_of_Fock_States]))
    circuit.append([Trotter_circuit]*Number_of_Trotter_Steps)
    circuit.append(cirq.measure(*qubits, key='m'))

    return circuit

_QUBIT_METRICS = ['sq_rb_pauli_error', 'zero_error', 'one_error']
_EDGE_METRIC = 'cz_inferred_gate_error_pauli'
_DEFAULT_WEIGHTS = {'sq_rb_pauli_error': 1, 'cz_inferred_gate_error_pauli': 1, 'zero_error': 1, 'one_error': 1}

def Load_Calibration(Project_id, Processor_id, Grid_Configuration, Snapshot=None):  # Latest snapshot if Snapshot is None
    processor = cirq_google.get_engine(Project_id).get_processor(Processor_id)
    config = processor.get_config(device_config_revision=cirq_google.Snapshot(Snapshot) if Snapshot else None, config_name=Grid_Configuration)
    print(f'Loaded calibration snapshot {config.snapshot_id}')
    return config.calibration, config.snapshot_id

def Logical_Interaction_Tree(Number_of_Bosonic_Modes, Number_of_Fock_States):  # (logical qubit, already-placed logical neighbour), in placement order
    Number_of_Qubits = Number_of_Bosonic_Modes * (Number_of_Fock_States+1)
    Spin_Qubits = range(Number_of_Bosonic_Modes*Number_of_Fock_States, Number_of_Qubits)
    Order = []
    for j in range(Number_of_Bosonic_Modes):
        Order.append((Spin_Qubits[j], Spin_Qubits[j-1] if j > 0 else None))
        Order.append(((j+1)*Number_of_Fock_States-1, Spin_Qubits[j]))
        for i in range(Number_of_Fock_States-2, -1, -1):
            Order.append((j*Number_of_Fock_States+i, j*Number_of_Fock_States+i+1))
    return Order, Spin_Qubits

def Mapping_Error(calibration, Qubit_Mapping, Number_of_Bosonic_Modes, Number_of_Fock_States, Weights=None):  # The cost minimised by Optimised_Qubit_Mapping
    Weights = _DEFAULT_WEIGHTS | (Weights or {})
    Order, _ = Logical_Interaction_Tree(Number_of_Bosonic_Modes, Number_of_Fock_States)
    Qubit_Errors = {key: {q[0]: float(v[0]) for q, v in calibration[key].items()} for key in _QUBIT_METRICS}
    Edge_Errors = {frozenset(q): float(v[0]) for q, v in calibration[_EDGE_METRIC].items()}
    Qubit_Cost = sum(Weights[key]*Qubit_Errors[key].get(q, np.nan) for key in _QUBIT_METRICS for q in Qubit_Mapping)
    Edge_Cost = sum(Weights[_EDGE_METRIC]*Edge_Errors.get(frozenset((Qubit_Mapping[logical], Qubit_Mapping[parent])), np.nan) for logical, parent in Order[1:])
    return Qubit_Cost + Edge_Cost

def Optimised_Qubit_Mapping(calibration, Number_of_Bosonic_Modes, Number_of_Fock_States, Weights=None, Cutoff_Step=1):
    Weights = _DEFAULT_WEIGHTS | (Weights or {})
    Qubit_Metrics = _QUBIT_METRICS
    Edge_Metric = _EDGE_METRIC

    Qubit_Errors = {key: {q[0]: float(v[0]) for q, v in calibration[key].items() if np.isfinite(v[0])} for key in Qubit_Metrics}
    Edge_Errors = {frozenset(q): float(v[0]) for q, v in calibration[Edge_Metric].items() if np.isfinite(v[0])}  # Uncalibrated couplers are NaN

    #---Logical interaction graph (a tree), in placement order---#

    Number_of_Qubits = Number_of_Bosonic_Modes * (Number_of_Fock_States+1)
    Order, _ = Logical_Interaction_Tree(Number_of_Bosonic_Modes, Number_of_Fock_States)
    Children = {logical: [] for logical, _ in Order}
    for logical, parent in Order[1:]:
        Children[parent].append(logical)

    #---Exclude the worst Cutoffs[key]% of qubits and couplers for each metric---#

    def Below_Cutoff(errors, key, Cutoffs):
        Threshold = np.percentile(list(errors.values()), 100 - Cutoffs[key])
        return {k for k, e in errors.items() if e <= Threshold}

    def Filter(Cutoffs):
        Good_Qubits = set.intersection(*(Below_Cutoff(Qubit_Errors[key], key, Cutoffs) for key in Qubit_Metrics))
        Good_Edges = {edge for edge in Below_Cutoff(Edge_Errors, Edge_Metric, Cutoffs) if edge <= Good_Qubits}
        Good_Neighbours = {q: [] for q in Good_Qubits}
        for edge in Good_Edges:
            q0, q1 = sorted(edge)
            Good_Neighbours[q0].append(q1)
            Good_Neighbours[q1].append(q0)
        return Good_Qubits, Good_Neighbours

    def Count_Mappings(Good_Neighbours):  # Non-backtracking tree homomorphisms: an upper bound on the number of valid mappings
        Arcs = [(p, q) for p in Good_Neighbours for q in Good_Neighbours[p]]
        Ways = {}  # Ways[logical][(p, q)]: embeddings of logical's subtree with logical on q and its parent on p
        for logical, parent in reversed(Order):
            Into = lambda q, p: math.prod(sum(Ways[c][(q, n)] for n in Good_Neighbours[q] if n != p) for c in Children[logical])
            if parent is None:
                return sum(Into(q, None) for q in Good_Neighbours)
            Ways[logical] = {(p, q): Into(q, p) for p, q in Arcs}

    #---Exhaustive search over embeddings of the (tree) interaction graph---#

    def Search(Good_Qubits, Good_Neighbours):
        Qubit_Cost = {q: sum(Weights[key]*Qubit_Errors[key][q] for key in Qubit_Metrics) for q in Good_Qubits}
        Edge_Cost = lambda a, b: Weights[Edge_Metric]*Edge_Errors[frozenset((a, b))]
        Min_Qubit_Cost = min(Qubit_Cost.values())
        Min_Edge_Cost = Weights[Edge_Metric]*min(Edge_Errors.values())
        Best = {'Cost': np.inf, 'Placement': None}
        Placement = {}

        def Place(k, Cost):
            Lower_Bound = Cost + (len(Order)-k)*Min_Qubit_Cost + (len(Order)-max(k, 1))*Min_Edge_Cost
            if Lower_Bound >= Best['Cost']:
                return
            if k == len(Order):
                Best['Cost'], Best['Placement'] = Cost, dict(Placement)
                return
            logical, parent = Order[k]
            Used = set(Placement.values())
            for q in (sorted(Good_Qubits) if parent is None else Good_Neighbours[Placement[parent]]):
                if q in Used:
                    continue
                Placement[logical] = q
                Place(k+1, Cost + Qubit_Cost[q] + (Edge_Cost(q, Placement[parent]) if parent is not None else 0))
                del Placement[logical]

        Place(0, 0.0)
        return Best

    #---Loosen cutoffs (step scaled by weight) from 100% until a mapping exists; take the optimal mapping at that level---#

    Steps = {key: Cutoff_Step * min(Weights.values()) / Weights[key] for key in Weights}  # Higher weight: loosened more slowly
    Round = 1
    while True:
        Cutoffs = {key: max(100 - Round*Steps[key], 0) for key in Weights}
        Good_Qubits, Good_Neighbours = Filter(Cutoffs)
        if len(Good_Qubits) >= Number_of_Qubits and Count_Mappings(Good_Neighbours) > 0:  # Cheap checks before the exhaustive search
            print(f'Searching at cutoffs {Cutoffs}')
            Best = Search(Good_Qubits, Good_Neighbours)
            if Best['Placement'] is not None:
                print(f'Optimal mapping found at cutoffs {Cutoffs}, weighted error = {Best["Cost"]:.4f}')
                break
        if all(v == 0 for v in Cutoffs.values()):
            raise ValueError(f'No qubit mapping exists on this snapshot: {Number_of_Qubits} connected qubits are needed, '
                             f'but no such chain exists even with no cutoffs.')
        Round += 1

    Qubit_Mapping = [Best['Placement'][k] for k in range(Number_of_Qubits)]

    Excluded_Qubits = [q[0] for q in calibration[Qubit_Metrics[0]] if q[0] not in Good_Qubits]
    Excluded_Edges = [q for q in calibration[Edge_Metric] if frozenset(q) not in Below_Cutoff(Edge_Errors, Edge_Metric, Cutoffs)]

    return Qubit_Mapping, Excluded_Qubits, Excluded_Edges

def Plot_Qubit_Mapping(calibration, Qubit_Mapping, Number_of_Bosonic_Modes, Number_of_Fock_States, Title, Excluded_Qubits=(), Excluded_Edges=()):  # The mapping on each error heatmap
    Order, Spin_Qubits = Logical_Interaction_Tree(Number_of_Bosonic_Modes, Number_of_Fock_States)
    Qubit_Mapping_Figure, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, key in zip(axes.flat, ['sq_rb_pauli_error', _EDGE_METRIC, 'zero_error', 'one_error']):
        calibration.heatmap(key).plot(ax, annotation_format=None)
        for logical, parent in Order[1:]:
            a, b = Qubit_Mapping[logical], Qubit_Mapping[parent]
            colour = 'purple' if (logical in Spin_Qubits) != (parent in Spin_Qubits) else ('blue' if logical in Spin_Qubits else 'red')
            ax.plot([a.col, b.col], [a.row, b.row], color=colour, lw=4)
        for k, q in enumerate(Qubit_Mapping):
            ax.plot(q.col, q.row, 'o', mfc='none', mec='blue' if k in Spin_Qubits else 'red', mew=2, ms=8)
        ax.plot([q.col for q in Excluded_Qubits], [q.row for q in Excluded_Qubits], 'x', color='grey', ms=6)
        ax.plot([(a.col+b.col)/2 for a, b in Excluded_Edges], [(a.row+b.row)/2 for a, b in Excluded_Edges], 'x', color='grey', ms=4)
    Qubit_Mapping_Figure.suptitle(f'{Title}, weighted error = {Mapping_Error(calibration, Qubit_Mapping, Number_of_Bosonic_Modes, Number_of_Fock_States):.4f}\n'
                                  f'(red: boson, blue: spin, purple: spin-boson, ×: excluded qubit / coupler above CZ cutoff)')
    Qubit_Mapping_Figure.tight_layout()

    return Qubit_Mapping_Figure

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

def Pauli_Twirl(circuits, Pauli_Twirling_Instances, Number_of_Shots):
    Twirled_Circuits = []
    for circuit in circuits:
        for twirl in range(Pauli_Twirling_Instances):
            Twirled = cirq.transformers.CZGaugeTransformer(circuit)
            Twirled = cirq.merge_single_qubit_moments_to_phxz(Twirled)
            Twirled_Circuits.append(cirq.drop_empty_moments(Twirled))
    Shots_Per_Twirl = Number_of_Shots//Pauli_Twirling_Instances
    return Twirled_Circuits, Shots_Per_Twirl

def Job_To_Quantum_Engine(Project_id, Processor_id, circuits, Number_of_Shots, Number_of_Fock_States, Number_of_Bosonic_Modes, Displacement_Coefficent, Spin_Interaction_Coefficent, Number_of_Trotter_Steps, Total_time, Pauli_Twirling_Instances):
    program_id = f'N{Number_of_Fock_States}L{Number_of_Bosonic_Modes}D{Displacement_Coefficent}J{Spin_Interaction_Coefficent}T{Total_time}Trot{Number_of_Trotter_Steps}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'.replace('.', 'p')
    job_id = f'Shots{Number_of_Shots}Twirls{Pauli_Twirling_Instances}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    engine = cirq_google.get_engine(Project_id)
    program = engine.create_program(circuits, program_id)
    program.run_sweep(Processor_id, job_id= job_id, repetitions=Number_of_Shots)
    return program_id, job_id

def UnaryPostSelection(results, Number_of_Fock_States, Number_of_Bosonic_Modes):
    Boson_Results = results[:, :Number_of_Bosonic_Modes*Number_of_Fock_States].reshape(-1, Number_of_Bosonic_Modes, Number_of_Fock_States)     # (shots, modes, Fock states)
    Spin_Results = results[:, Number_of_Bosonic_Modes*Number_of_Fock_States:]     # (shots, modes): spin j is attached to mode j
    Unary_Mask = Boson_Results.sum(axis=2) == 1          # (shots, modes): True if that register has exactly one qubit in |1>, also masks the attached spin
    return Boson_Results, Spin_Results, Unary_Mask