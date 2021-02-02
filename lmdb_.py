import lmdb
import pickle

def store(root, object_, idx, map_size):
    dir_ = root + '/lmdb'
    env = lmdb.open(dir_, map_size=map_size)
    with env.begin(write=True) as txn:
        txn.put(f'idx-{idx}'.encode('ascii'), pickle.dumps(object_))
    env.close()


def read(root, idx):
    dir_ = root + '/lmdb'
    env = lmdb.open(dir_, map_size=map_size)
    sample = None
    with env.begin(write=False) as txn:
        sample = txn.get(f'idx-{idx}'.encode('ascii'))
    env.close()
    return sample
