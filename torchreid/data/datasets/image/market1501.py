from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings
from ..dataset import ImageDataset

class Market1501(ImageDataset):
    """
    Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).

    Dataset structure:
        reid-data/
        └── market1501/
            ├── bounding_box_train/
            ├── bounding_box_test/
            ├── query/
            ├── gt_bbox/
            ├── gt_query/
            └── other files...

    """

    _junk_pids = [0, -1]
    dataset_dir = r'C:\\reid\deep-person-reid\\reid-data\\market1501'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        # Use the fixed dataset path
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # Paths to specific subdirectories
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.dataset_dir, 'gt_bbox')  # Optional, use if needed
        self.gt_query_dir = osp.join(self.dataset_dir, 'gt_query')  # Optional, use if needed
        self.market1501_500k = market1501_500k

        # Ensure all required directories exist
        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        # Process the data
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
