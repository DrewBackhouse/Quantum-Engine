import cirq

from Functions import Trotter_Step_Circuit, Full_Circuit, Compile_For_Willow, Sample_On_Quantum_Engine, UnaryPostSelection

#---Model Parameters---#

Number_of_Fock_States = 8
Number_of_Bosonic_Modes = 1
Displacement_Coefficent = 1
Spin_Interaction_Coefficent = 0.1
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip

#---Simulation Parameters---#

Total_time = 25
Number_of_Trotter_Steps = 1
Number_of_Shots = 10
Qubit_Mapping = [cirq.GridQubit(6, 1), cirq.GridQubit(6, 2), cirq.GridQubit(5, 2), cirq.GridQubit(4, 2), cirq.GridQubit(4, 3), cirq.GridQubit(3, 3), cirq.GridQubit(3, 4), cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)]

#---Circuit Generation---#

Trotter_Time = Total_time/Number_of_Trotter_Steps
Trotter_Circuit, qubits = Trotter_Step_Circuit(Number_of_Fock_States, Number_of_Bosonic_Modes, Trotter_Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
print(Trotter_Circuit)
Full_Trotterised_Circuit = Full_Circuit(Trotter_Circuit, qubits, Number_of_Bosonic_Modes, Number_of_Fock_States, Number_of_Trotter_Steps)
print(Full_Trotterised_Circuit)
Full_Trotterised_Circuit_Stratified = Compile_For_Willow(Full_Trotterised_Circuit, qubits, Qubit_Mapping)
print(Full_Trotterised_Circuit_Stratified.to_text_diagram(qubit_order=Qubit_Mapping))

#---Circuit Sampling---#

if __name__ == '__main__':

    from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook
    qcs_objects = get_qcs_objects_for_notebook(project_id='quantum-neuron-network', processor_id='WLA1HHPR00V07_5B_MINT')
    processor_id = qcs_objects.processor_id
    engine = qcs_objects.engine

    results = Sample_On_Quantum_Engine(Full_Trotterised_Circuit_Stratified, engine, processor_id, Number_of_Shots)
    results = results.measurements['m'].astype(int)
    print(results)

results_Postselected = UnaryPostSelection(results, Number_of_Fock_States, Number_of_Bosonic_Modes)
print(results_Postselected)