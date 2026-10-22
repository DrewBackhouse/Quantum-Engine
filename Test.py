# The Google Cloud Project id to use.
project_id = "quantum-neuron-network"  # @param {type:"string"}

from cirq_google.engine import get_engine
from cirq_google.engine.qcs_notebook import authenticate_user

try:
    engine = get_engine(project_id)
    print("sucsess")
except Exception:
    from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook

    print("Unable to connect to a hardware engine. Using a virtual instance instead.")
    qcs_objects = get_qcs_objects_for_notebook(project_id, virtual=True)
    engine = qcs_objects.engine