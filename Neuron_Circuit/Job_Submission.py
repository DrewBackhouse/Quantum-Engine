# Using conda venv 'cirq'

import numpy
import cirq
import os

from Functions import Trotter_Step_Circuit, Trotter_Step_Circuit_2nd_Order, Full_Circuit, Compile_For_Willow, Job_To_Quantum_Engine, Pauli_Twirl, Load_Calibration, Optimised_Qubit_Mapping, Plot_Qubit_Mapping


#---Model Parameters---#

Number_of_Fock_States = 8
Number_of_Bosonic_Modes = 3
Displacement_Coefficent = 1
Spin_Interaction_Coefficent = 0.1
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip


#---Simulation Parameters---#

Project_id= "quantum-neuron-network"
Processor_id = 'WLA1HHPR00V07_5B_MINT'
Total_time = 25
Number_of_Trotter_Steps = 44
Number_of_Shots = 2000
Grid_Configuration = 'cz_104'
Optimise_Qubit_Mapping = True
Manual_Qubit_Mapping = Manual_Qubit_Mapping = [cirq.GridQubit(4, 6), cirq.GridQubit(4, 7), cirq.GridQubit(4, 8), cirq.GridQubit(4, 9), cirq.GridQubit(4, 10), cirq.GridQubit(5, 10), cirq.GridQubit(6, 10), cirq.GridQubit(7, 10), cirq.GridQubit(10, 7), cirq.GridQubit(10, 8), cirq.GridQubit(11, 8), cirq.GridQubit(11, 9), cirq.GridQubit(10, 9), cirq.GridQubit(10, 10), cirq.GridQubit(9, 10), cirq.GridQubit(9, 9), cirq.GridQubit(9, 5), cirq.GridQubit(9, 6), cirq.GridQubit(9, 7), cirq.GridQubit(8, 7), cirq.GridQubit(8, 6), cirq.GridQubit(7, 6), cirq.GridQubit(7, 7), cirq.GridQubit(7, 8), cirq.GridQubit(7, 9), cirq.GridQubit(8, 9), cirq.GridQubit(8, 8)]
Pauli_Twirling_Instances = 10
Submit_To_Quantum_Engine = True

#---Qubit Mapping---#

calibration, snapshot_id = Load_Calibration(Project_id, Processor_id, Grid_Configuration, Snapshot=None)
if Optimise_Qubit_Mapping:
    Qubit_Mapping, Excluded_Qubits, Excluded_Edges = Optimised_Qubit_Mapping(calibration, Number_of_Bosonic_Modes, Number_of_Fock_States, Weights=None)
else:
    Qubit_Mapping, Excluded_Qubits, Excluded_Edges = Manual_Qubit_Mapping, [], []
Qubit_Mapping_Figure = Plot_Qubit_Mapping(calibration, Qubit_Mapping, Number_of_Bosonic_Modes, Number_of_Fock_States, f'{Processor_id} snapshot {snapshot_id}: L={Number_of_Bosonic_Modes}, N={Number_of_Fock_States}, {"optimised" if Optimise_Qubit_Mapping else "manual"} mapping', Excluded_Qubits, Excluded_Edges)
Qubit_Mapping_Figure.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Qubit_Mapping_Figure.png'), bbox_inches='tight')
input('Proceed with this mapping?')

#---Circuit Generation---#

Trotter_Time = Total_time/Number_of_Trotter_Steps
Trotter_Circuit, qubits = Trotter_Step_Circuit(Number_of_Fock_States, Number_of_Bosonic_Modes, Trotter_Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
print(Trotter_Circuit)
input('Proceed with this trotter circuit?')
circuits = []
for steps in range(Number_of_Trotter_Steps):
    print(f'Generating timestep circuit {steps}')
    Full_Trotterised_Circuit = Full_Circuit(Trotter_Circuit, qubits, Number_of_Bosonic_Modes, Number_of_Fock_States, steps +1)
    #print(Full_Trotterised_Circuit)
    Full_Trotterised_Circuit_Stratified = Compile_For_Willow(Full_Trotterised_Circuit, qubits, Qubit_Mapping)
    #print(Full_Trotterised_Circuit_Stratified.to_text_diagram(qubit_order=Qubit_Mapping))
    circuits.append(Full_Trotterised_Circuit_Stratified)

if Pauli_Twirling_Instances >0:
    print('Twirling circuits')
    circuits, Number_of_Shots = Pauli_Twirl(circuits, Pauli_Twirling_Instances, Number_of_Shots)

#---Job submission---#

if Submit_To_Quantum_Engine:
    program_id, job_id = Job_To_Quantum_Engine(Project_id, Processor_id, circuits, Number_of_Shots, Number_of_Fock_States, Number_of_Bosonic_Modes, Displacement_Coefficent, Spin_Interaction_Coefficent, Number_of_Trotter_Steps, Total_time, Pauli_Twirling_Instances)
    print(f"Submitted batch job: program_id = {program_id}, job_id = {job_id}")
    raise SystemExit

#---Simulator---#

