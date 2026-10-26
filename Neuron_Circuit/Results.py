import cirq_google
import numpy as np
import matplotlib.pyplot as plt

from Functions import UnaryPostSelection

#---Job details---#

Project_id = "quantum-neuron-network"
Program_id = 'N8L1D1J0p1T25Trot25-20261026-144029'
Job_id = 'Shots1000-20261026-144029'
Number_of_Fock_States = 8
Number_of_Bosonic_Modes = 1
Total_Time = 25
Number_of_Trotter_Steps = 25
Number_of_Shots = 1000

# Should encode this info in the measurement structure such that it does not need to be defined here.

#---Fetch results---#

engine = cirq_google.get_engine(Project_id) 
program = engine.get_program(Program_id)
job = program.get_job(Job_id)
results = job.results()
all_bits = np.stack([r.measurements['m'].astype(int) for r in results])
print(all_bits)
print(all_bits[0])

#---Post-processing of results---#

All_Results = []
Min_Kept = Number_of_Shots
for j in range(Number_of_Trotter_Steps):
    Full_Results_Postselected = UnaryPostSelection(all_bits[j], Number_of_Fock_States, Number_of_Bosonic_Modes)
    Min_Kept = min(Min_Kept, len(Full_Results_Postselected))
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

fig, (ax_boson, ax_spin) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

for j in range(Number_of_Bosonic_Modes):
    ax_boson.plot(Time_Data, Boson_Occupation_Data[:, j], marker='o', label=f'Mode {j}')
ax_boson.set_ylabel('Average occupation number')
ax_boson.legend()

for j in range(Spin_Data.shape[1]):
    ax_spin.plot(Time_Data, Spin_Data[:, j], marker='o', label=f'Spin {j}')
ax_spin.set_ylabel('Average spin state P(|1⟩)')
ax_spin.set_ylim(0, 1)
ax_spin.set_xlabel('Time')
ax_spin.legend()

plt.tight_layout()
plt.show()