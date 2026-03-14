import json
import os

from jsonschema import validate

SCHEMA_INSTANCE_PATH = "../salt-spreading/support/schema_instance.json"
SCHEMA_SOLUTION_ARCS_PATH = "../salt-spreading/support/schema_solution_arcs.json"
SCHEMA_SOLUTION_NODES_PATH = "../salt-spreading/support/schema_solution_nodes.json"


def load_instance(path: str) -> dict:
    if not os.path.exists(path):
        raise IOError("instance file does not exist")
    if not os.path.exists(SCHEMA_INSTANCE_PATH):
        raise IOError("schema instance not found")
    with open(path, "r") as file:
        with open(SCHEMA_INSTANCE_PATH, "r") as schema:
            instance = json.load(file)
            validate(instance, json.load(schema))  # will raise exception if not valid
            return instance
