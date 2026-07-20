from sage_lib.partition.Partition import Partition

def remove_redundancy_from_partition(partition: Partition) -> Partition:
    """
    Remove redundant elements from a partition by instancing a new partition and adding elements with duplicate filtering on.

    Args:
        partition (Partition): The input partition.
    """
    print(len(partition))
    print(partition[1].atoms.distance_matrix)
    hashes=partition.get_metadata('hash')
    
    print(hashes[0:10])

if __name__ == "__main__":
    partition=Partition(local_root='/home/jholl/alomancy_runs/sulfur_training/results/global_database/hybrid_local', stroage='hybrid', access='ro')
    print(len(partition))
    remove_redundancy_from_partition(partition)