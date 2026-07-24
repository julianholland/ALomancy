import numpy as np
from tqdm import tqdm


def scale_global_descriptor_length(
    normalized_char_vec: np.ndarray, dimensions: int = 0
) -> np.ndarray:
    """Embeds the characteristic distance vector into a fixed dimension."""
    if dimensions == 0 or dimensions is None:
        return normalized_char_vec

    if len(normalized_char_vec) == 0:
        return np.zeros(dimensions)
    if len(normalized_char_vec) < dimensions:
        # Interpolate to the desired dimension
        x_old = np.linspace(0, 1, len(normalized_char_vec))
        x_new = np.linspace(0, 1, dimensions)
        scaled_char_vec = np.interp(x_new, x_old, normalized_char_vec)
    else:
        # Downsample to the desired dimension
        indices = np.linspace(
            0, len(normalized_char_vec) - 1, dimensions
        ).astype(int)
        scaled_char_vec = normalized_char_vec[indices]

    return scaled_char_vec


def make_char_vec(structure, dimensions: int = 128) -> np.ndarray:
    """Returns the characteristic distance vector from a given ase atoms object

    Args:
        dimensions (int, optional): The desired dimensionality of the descriptor. Defaults to 128.

    Returns:
        np.ndarray: A normalised global descriptor
    """
    dist_mat = structure.distance_matrix
    upper = np.triu(dist_mat)
    flat = upper.flatten()
    no_zeroes = flat[flat != 0]  # Remove zero distances (self-distances)
    char_vec = np.sort(no_zeroes)
    scaled_char_vec = scale_global_descriptor_length(char_vec, dimensions=dimensions)

    return scaled_char_vec


def assign_descriptor_to_container(container, dimensions: int = 128) -> None:
    """Assigns the characteristic distance vector to a given container.

    Args:
        container (Container): The input container.
        dimensions (int, optional): The desired dimensionality of the descriptor. Defaults to 128.
    """
    char_vec = make_char_vec(container.atoms, dimensions=dimensions)
    container.atoms.metadata["char_vec"] = char_vec.tolist()


def assign_descriptor_to_all_partition(partition, dimensions: int = 128) -> None:
    """Assigns the characteristic distance vector to each structure in the partition.

    Args:
        partition (Partition): The input partition.
        dimensions (int, optional): The desired dimensionality of the descriptor. Defaults to 128.
    """
    partition_ram = list(partition.containers)

    for container in tqdm(
        partition_ram, desc="Assigning descriptors to partition", total=len(partition_ram)
    ):
        assign_descriptor_to_container(container, dimensions=dimensions)

    # Pass bare list so update_containers uses 0-based positional indices correctly.
    partition.update_containers(partition_ram)


if __name__ == "__main__":
    from sage_lib.partition.Partition import Partition

    partition_path = "/home/jholl/alomancy_runs/test_gdb"
    partition = Partition(local_root=partition_path, storage="hybrid", access="rw")
    assign_descriptor_to_all_partition(partition, dimensions=128)
    print(partition[1000].atoms.metadata["char_vec"])
