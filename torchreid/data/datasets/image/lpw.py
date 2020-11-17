import glob

import os.path as osp
from os import listdir

from ..dataset import ImageDataset


class LPW(ImageDataset):
    """
    Labeled Pedestrian in the Wild
    Reference:
    Guanglu Song, Biao Leng, Yu Liu, Congrui Hetang, Shaofan Cai
    Region-based Quality Estimation Network for Large-Scale Person Re-identiﬁcation
    URL: http://liuyu.us/dataset/lpw/index.html

    Dataset statistics:
    # cameras: 11
    # identities: 4587
    # images: 590547
    """
    dataset_dir = 'pep_256x128'

    def __init__(self, root='/data', verbose=True, **kwargs):
        self.dataset_dir = osp.join(osp.expanduser(root), self.dataset_dir)
        camid = 0
        train = []
        pid_map = {}

        for scen in sorted(listdir(self.dataset_dir)):
            spath = osp.join(self.dataset_dir, scen)
            # share pid among views
            if osp.isdir(spath) == False: continue
            for view in sorted(listdir(spath)):
                vpath = osp.join(spath, view)
                for p in sorted(listdir(vpath)):
                    pid = pid_map.setdefault((scen, p), len(pid_map))
                    ppath = osp.join(vpath, p)
                    for im in sorted(listdir(ppath)):
                        ipath = osp.join(ppath, im)
                        train.append((ipath, pid, camid))
                camid += 1

        query = []
        gallery = []
        super().__init__(train, query, gallery, **kwargs)


if __name__ == '__main__':
    lpw = LPW()
