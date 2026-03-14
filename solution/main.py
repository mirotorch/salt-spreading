from jsonschema import ValidationError

from solution.io import load_instance

if __name__ == "__main__":
    problem = None
    try:
        problem = load_instance("../salt-spreading/data/belben/belben.json")
    except ValidationError as e:
        print("failed to validate input: " + str(e))
    except IOError as e:
        print(e.strerror)

