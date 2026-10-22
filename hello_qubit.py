# Set up Google Cloud authentication: gcloud auth application-default login

import cirq
import cirq_google as cg


def example(engine, processor_id):
    """Hello qubit example run against a quantum processor."""

    # Define a qubit on the device.
    processor = engine.get_processor(processor_id)
    qubit = sorted(processor.get_device().metadata.qubit_set)[0]

    # Create a circuit (qubits start in the |0> state).
    circuit = cirq.Circuit(
        cirq.X(qubit) ** 0.5,  # Square root of NOT.
        cirq.measure(qubit, key='result'),  # Measurement.
    )

    sampler = processor.get_sampler()
    print("Sampling circuit on Quantum Engine...\n")
    results = sampler.run(circuit, repetitions=1000)

    print("\nMeasurement results:")
    print(results)


if __name__ == '__main__':
    # Set up QCS objects.
    from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook

    qcs_objects = get_qcs_objects_for_notebook(project_id='quantum-neuron-network')
    processor_id = qcs_objects.processor_id
    engine = qcs_objects.engine

    example(engine, processor_id)