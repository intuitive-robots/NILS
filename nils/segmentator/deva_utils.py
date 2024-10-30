from typing import List, Optional

import numpy as np
from deva.utils.pano_utils import id_to_rgb
from scipy import stats


class ObjectInfo:
    """Stores meta information for an object
    """
    def __init__(self,
                 id: int,
                 category_id: Optional[int] = None,
                 isthing: Optional[bool] = None,
                 score: Optional[float] = None,
                 class_scores: Optional[List[float]] = None):
        self.id = id
        self.category_ids = [category_id]
        self.scores = [score]

        if class_scores is None:
            self.class_scores = []
        else:
            self.class_scores = class_scores
        
        self.isthing = isthing
        self.poke_count = 0  # number of detections since last this object was last seen

        self.missed_merges = 0
        self.missed_concensous = []

        self.preds = 0


    def poke(self) -> None:
        self.poke_count += 1

    def unpoke(self) -> None:
        self.poke_count = 0

    def merge(self, other) -> None:
        self.category_ids.extend(other.category_ids)
        self.scores.extend(other.scores)
        
        self.class_scores.extend(other.class_scores)

        self.missed_merges += other.missed_merges
        self.missed_concensous.extend(other.missed_concensous)
        self.preds += other.preds

    def del_class_id(self, class_id: int) -> None:

        self.class_scores = [s for i, s in enumerate(self.class_scores) if self.category_ids[i] != class_id]
        self.scores = [s for i, s in enumerate(self.scores) if self.category_ids[i] != class_id]
        self.category_ids = [c for c in self.category_ids if c != class_id]
    

    def vote_category_id_scored(self) -> Optional[int]:
        category_ids = [c for c in self.category_ids if c is not None]
        category_scores = {c: 0 for c in category_ids}
        for c, s in zip(self.category_ids, self.class_scores):
            category_scores[c] += s

        max_id = max(category_scores, key=category_scores.get)


        return max_id

    def vote_category_id(self) -> Optional[int]:
        category_ids = [c for c in self.category_ids if c is not None]
        if len(category_ids) == 0:
            return None
        else:
            return int(stats.mode(category_ids, keepdims=False)[0])

    def vote_score(self) -> Optional[float]:
        scores = [c for c in self.scores if c is not None]
        if len(scores) == 0:
            return None
        else:
            return float(np.mean(scores))

    def get_rgb(self) -> np.ndarray:
        # this is valid for panoptic segmentation-style id only (0~255**3)
        return id_to_rgb(self.id)

    def copy_meta_info(self, other) -> None:
        self.category_ids = other.category_ids
        self.scores = other.scores
        self.isthing = other.isthing
        self.class_scores = other.class_scores

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f'(ID: {self.id}, cat: {self.category_ids}, isthing: {self.isthing}, score: {self.scores}, class_scores: {self.class_scores})'
