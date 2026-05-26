def run(args):
    name = args.get("name", "World")
    return f"Hello, {name}!"

def get_info():
    return {
        "name": "Greeter",
        "description": "Returns a greeting message. Accepts a 'name' argument."
    }