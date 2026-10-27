import os
import re
import cirq
import cirq_google
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Functions import UnaryPostSelection

#---Job details---#

Project_id = "quantum-neuron-network"
Program_id = 'N8L1D1J0p1T25Trot100-20261027-120235'
Job_id = 'Shots2000-20261027-120235'

Parameters = re.match(r'N(\d+)L(\d+)D([\dp]+)J([\dp]+)T([\dp]+)Trot(\d+)-', Program_id)
Number_of_Fock_States = int(Parameters.group(1))
Number_of_Bosonic_Modes = int(Parameters.group(2))
Total_Time = float(Parameters.group(5).replace('p', '.'))
Number_of_Trotter_Steps = int(Parameters.group(6))

#---Fetch results---#

engine = cirq_google.get_engine(Project_id) 
program = engine.get_program(Program_id)
circuits = program.get_circuits()
job = program.get_job(Job_id)
results = job.results()
all_bits = np.stack([r.measurements['m'].astype(int) for r in results])
assert len(results) == Number_of_Trotter_Steps, f'Expected {Number_of_Trotter_Steps} circuits, job returned {len(results)}'

#---Post-processing of results---#

All_Results = []
Survival_Rate = []
for j in range(Number_of_Trotter_Steps):
    Full_Results_Postselected = UnaryPostSelection(all_bits[j], Number_of_Fock_States, Number_of_Bosonic_Modes)
    Survival_Rate.append(len(Full_Results_Postselected) / len(all_bits[j]))
    Averaged_Results = Full_Results_Postselected.mean(axis=0)
    Boson_Results = Averaged_Results[:Number_of_Bosonic_Modes*Number_of_Fock_States].reshape(Number_of_Bosonic_Modes, Number_of_Fock_States)
    Boson_Occupation_Number = np.sum(Boson_Results * np.arange(0, Number_of_Fock_States, 1), axis=1)
    Timestep_Results = np.concatenate([Boson_Occupation_Number, Averaged_Results[Number_of_Bosonic_Modes*Number_of_Fock_States:]])
    All_Results.append(Timestep_Results)

#---Plots---#

Time_Data = np.linspace(Total_Time/Number_of_Trotter_Steps, Total_Time, Number_of_Trotter_Steps) # Circuit j contains j+1 Trotter steps

All_Results = np.array(All_Results)
Boson_Occupation_Data = All_Results[:, :Number_of_Bosonic_Modes]
Spin_Data = All_Results[:, Number_of_Bosonic_Modes:]     # Probability each spin qubit is in |1>

Qubit_Order = next(op for op in circuits[0].all_operations() if cirq.is_measurement(op)).qubits  # Measurement keeps the logical qubit order
Circuit_Text = circuits[0].to_text_diagram(qubit_order=Qubit_Order)
Circuit_Lines = Circuit_Text.splitlines()
fig_circuit = plt.figure(figsize=(max(8, 0.1*max(len(l) for l in Circuit_Lines)), max(2, 0.2*len(Circuit_Lines))))
fig_circuit.text(0.01, 0.99, 'Circuit for first timestep\n\n' + Circuit_Text, family='monospace', fontsize=8, va='top', ha='left')

fig, (ax_boson, ax_spin, ax_survival) = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

for j in range(Number_of_Bosonic_Modes):
    ax_boson.plot(Time_Data, Boson_Occupation_Data[:, j], marker='o', label=f'Mode {j}')
ax_boson.set_ylabel('Average occupation number')
ax_boson.legend()

for j in range(Spin_Data.shape[1]):
    ax_spin.plot(Time_Data, Spin_Data[:, j], marker='o', label=f'Spin {j}')
ax_spin.set_ylabel('Average spin state P(|1⟩)')
ax_spin.set_ylim(0, 1)
ax_spin.legend()

ax_survival.plot(Time_Data, Survival_Rate, marker='o', color='k')
ax_survival.set_ylabel('Post-selection survival rate')
ax_survival.set_ylim(0, 1)
ax_survival.set_xlabel('Time')

plt.tight_layout()

with PdfPages(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures', f'{Program_id}_{Job_id}.pdf')) as pdf:
    pdf.savefig(fig_circuit, bbox_inches='tight')
    pdf.savefig(fig, bbox_inches='tight')

plt.show()