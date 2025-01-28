from src.System import Consys
from src.Controller import PIDController
from src.Plant import BathtubPlant, CournotPlant
import yaml
from jax.tree_util import register_pytree_node


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    system = Consys(config["pivitol_parameters"])

    system.run() 
    


if __name__ == "__main__":
    main()