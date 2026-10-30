import os
import re
import cirq
import cirq_google
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Functions import UnaryPostSelection, Plot_Qubit_Mapping


#---Job details---#

Project_id = "quantum-neuron-network"
Program_id = 'N8L3D1J0p1T25Trot44-20261030-160946'
Job_id = 'Shots200Twirls10-20261030-160946'

Parameters = re.match(r'N(\d+)L(\d+)D([\dp]+)J([\dp]+)T([\dp]+)Trot(\d+)-', Program_id)
Twirl_Match = re.match(r'Shots(\d+)Twirls(\d+)-', Job_id)
Number_of_Fock_States = int(Parameters.group(1))
Number_of_Bosonic_Modes = int(Parameters.group(2))
Total_Time = float(Parameters.group(5).replace('p', '.'))
Number_of_Trotter_Steps = int(Parameters.group(6))
Pauli_Twirling_Instances = int(Twirl_Match.group(2)) if Twirl_Match else 0
Circuits_Per_Timestep = max(1, Pauli_Twirling_Instances)


#---Fetch results---#

engine = cirq_google.get_engine(Project_id) 
program = engine.get_program(Program_id)
circuits = program.get_circuits()
job = program.get_job(Job_id)
#print(job.failure())
#input()
results = job.results()
all_bits = np.stack([r.measurements['m'].astype(int) for r in results])
assert len(results) == Number_of_Trotter_Steps * Circuits_Per_Timestep, f'Expected {Number_of_Trotter_Steps * Circuits_Per_Timestep} circuits, job returned {len(results)}'
all_bits = all_bits.reshape(Number_of_Trotter_Steps, -1, all_bits.shape[-1])


#---Post-processing of results---#

All_Results = []
Survival_Rate = []          # Per mode (and its attached spin): fraction of shots where that register is unary
All_Valid_Rate = []         # Fraction of shots where every register is unary
Valid_Shot_Counts = []
for j in range(Number_of_Trotter_Steps):
    Boson_Results, Spin_Results, Unary_Mask = UnaryPostSelection(all_bits[j], Number_of_Fock_States, Number_of_Bosonic_Modes)
    Valid_Shots = Unary_Mask.sum(axis=0)     # (modes,) shots kept per mode
    Valid_Shot_Counts.append(Valid_Shots)
    Survival_Rate.append(Valid_Shots / len(all_bits[j]))
    All_Valid_Rate.append(np.all(Unary_Mask, axis=1).mean())
    Occupation_Per_Shot = Boson_Results @ np.arange(Number_of_Fock_States)     # (shots, modes)
    with np.errstate(invalid='ignore', divide='ignore'):
        Boson_Occupation_Number = (Occupation_Per_Shot * Unary_Mask).sum(axis=0) / Valid_Shots     # Average over each mode's valid shots only, NaN if none
        Spin_Probability = (Spin_Results * Unary_Mask).sum(axis=0) / Valid_Shots     # Spin j averaged over mode j's valid shots only
    Timestep_Results = np.concatenate([Boson_Occupation_Number, Spin_Probability])
    All_Results.append(Timestep_Results)
Survival_Rate = np.array(Survival_Rate)
Valid_Shot_Counts = np.array(Valid_Shot_Counts)


#---Plots---#

Time_Data = np.linspace(Total_Time/Number_of_Trotter_Steps, Total_Time, Number_of_Trotter_Steps) # Circuit j contains j+1 Trotter steps

All_Results = np.array(All_Results)
Boson_Occupation_Data = All_Results[:, :Number_of_Bosonic_Modes]
Spin_Data = All_Results[:, Number_of_Bosonic_Modes:]     # Probability each spin qubit is in |1>

Qubit_Order = next(op for op in circuits[0].all_operations() if cirq.is_measurement(op)).qubits  # Measurement keeps the logical qubit order
try:
    config = job.get_config()  # Calibration snapshot the job actually ran with
except ValueError as e:
    config = None
    print(f'No calibration for job ({e})')
if config is not None:
    fig_mapping = Plot_Qubit_Mapping(config.calibration, list(Qubit_Order), Number_of_Bosonic_Modes, Number_of_Fock_States, f'{config.processor_id} snapshot {config.snapshot_id}: L={Number_of_Bosonic_Modes}, N={Number_of_Fock_States}')
else:
    fig_mapping = None
    print('Skipping qubit mapping figure')

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

for j in range(Number_of_Bosonic_Modes):
    ax_survival.plot(Time_Data, Survival_Rate[:, j], marker='o', label=f'Mode {j}')
ax_survival.plot(Time_Data, All_Valid_Rate, linestyle='--', color='k', label='All modes')
ax_survival.set_ylabel('Post-selection survival rate')
ax_survival.set_ylim(0, 1)
ax_survival.legend()
ax_survival.set_xlabel('Time')

fig.suptitle(f'L={Number_of_Bosonic_Modes}, N={Number_of_Fock_States}, {Number_of_Trotter_Steps} Trotter steps')
plt.tight_layout()

Program_Name = re.sub(r'-\d{8}-\d{6}$', '', Program_id)
with PdfPages(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures', f'{Program_Name}_{Job_id}_NewPostSelection.pdf')) as pdf:
    if fig_mapping is not None:
        pdf.savefig(fig_mapping, bbox_inches='tight')
    pdf.savefig(fig_circuit, bbox_inches='tight')
    pdf.savefig(fig, bbox_inches='tight')