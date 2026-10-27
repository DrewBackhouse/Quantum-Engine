import numpy
import cirq

from Functions import Trotter_Step_Circuit, Full_Circuit, Compile_For_Willow, Sample_On_Quantum_Engine, UnaryPostSelection, Job_To_Quantum_Engine


#---Model Parameters---#

Number_of_Fock_States = 8
Number_of_Bosonic_Modes = 1
Displacement_Coefficent = 1
Spin_Interaction_Coefficent = 0.1
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip


#---Simulation Parameters---#

Total_time = 25
Number_of_Trotter_Steps = 70
Number_of_Shots = 2000
Project_id= "quantum-neuron-network"
Processor_id = 'WLA1HHPR00V07_5B_MINT'
Grid_Configuration = 'cz_104'
Qubit_Mapping = [cirq.GridQubit(6, 1), cirq.GridQubit(6, 2), cirq.GridQubit(5, 2), cirq.GridQubit(4, 2), cirq.GridQubit(4, 3), cirq.GridQubit(3, 3), cirq.GridQubit(3, 4), cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)]


#---Circuit Generation---#

Trotter_Time = Total_time/Number_of_Trotter_Steps
Trotter_Circuit, qubits = Trotter_Step_Circuit(Number_of_Fock_States, Number_of_Bosonic_Modes, Trotter_Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
print(Trotter_Circuit)
circuits = []
for steps in range(Number_of_Trotter_Steps):
    Full_Trotterised_Circuit = Full_Circuit(Trotter_Circuit, qubits, Number_of_Bosonic_Modes, Number_of_Fock_States, steps +1)
    #print(Full_Trotterised_Circuit)
    Full_Trotterised_Circuit_Stratified = Compile_For_Willow(Full_Trotterised_Circuit, qubits, Qubit_Mapping)
    #print(Full_Trotterised_Circuit_Stratified.to_text_diagram(qubit_order=Qubit_Mapping))
    circuits.append(Full_Trotterised_Circuit_Stratified)

#---Job submission---#

program_id, job_id = Job_To_Quantum_Engine(Project_id, Processor_id, circuits, Number_of_Shots, Number_of_Fock_States, Number_of_Bosonic_Modes, Displacement_Coefficent, Spin_Interaction_Coefficent, Number_of_Trotter_Steps, Total_time)
print(f"Submitted batch job: program_id = {program_id}, job_id = {job_id}")


#---Circuit Sampling---#

#results = Sample_On_Quantum_Engine(Project_id, Processor_id, Grid_Configuration, circuits, Number_of_Shots, Number_of_Trotter_Steps)
#for j in range(Number_of_Trotter_Steps):
    #print(results[j][0].measurements['m'].astype(int))
#input()

#results_Postselected = UnaryPostSelection(results, Number_of_Fock_States, Number_of_Bosonic_Modes)
#print(results_Postselected)