from abc import abstractmethod
import numpy as np
import torch


class FeatureSelectionAgorithm():
    """Base class for feature selection algorithms. Based on the implementation here: http://enggjournals.com/ijcse/doc/IJCSE11-03-05-051.pdf"""
    def __init__(self, dataset, predict_score_method, prediction_args, features_dim=2):
        self.dataset = dataset
        self.num_features = dataset.shape[features_dim]
        self.all_feature_set = set(range(self.num_features))
        self.predict_score = predict_score_method
        self.prediction_args = prediction_args
        self.features_used = set()
        
    def add_one_feature(self, existing_dataset, featureid, dim=2):
        if dim != 2:
            raise NotImplementedError
        
        new_tensor = existing_dataset.clone()
        new_tensor[:, :, featureid] = self.dataset[:, :, featureid]
        return new_tensor

    def remove_one_feature(self, existing_dataset, featureid, dim=2):
        if dim != 2:
            raise NotImplementedError

        new_tensor = existing_dataset.clone()
        new_tensor[:, :, featureid] = 0
        return new_tensor

    def find_feature_to_add(self, existing_dataset, feature_set_to_add):
        scores = []
        tensors = []
        feature_set_to_add_list = list(feature_set_to_add)
        # print("feature_set_to_add_list", feature_set_to_add_list)
        for featureid in feature_set_to_add_list:
            new_tensor = self.add_one_feature(existing_dataset, featureid)
            score = self.predict_score(new_tensor, **self.prediction_args)

            scores.append(score)
            tensors.append(new_tensor)
            # print(f"featureid: {featureid}, score: {score}")
        
        best_feature_ind = np.argmax(scores)  # assuming we want to maximize the score and not minimize it

        best_feature = feature_set_to_add_list[best_feature_ind]
        best_score = scores[best_feature_ind]
        best_dataset = tensors[best_feature_ind]
        return best_score, best_feature, best_dataset

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


class LRS(FeatureSelectionAgorithm):
    """Implements LRS: Plus-L-Minus-R Selection. Based on the implementation here: http://enggjournals.com/ijcse/doc/IJCSE11-03-05-051.pdf"""
    def __init__(self, dataset, predict_score_method, l, r, features_dim=2, **prediction_args):
        super().__init__(dataset, predict_score_method, features_dim, **prediction_args)
        self.l = l
        self.r = r


    def find_feature_to_remove(self, existing_dataset):
        scores = []
        tensors = []
        features_used_list = list(self.features_used)
        for featureid in features_used_list:
            new_tensor = self.remove_one_feature(existing_dataset, featureid)

            score = self.predict_score(new_tensor, **self.prediction_args)
            scores.append(score)
            tensors.append(new_tensor)
        
        best_feature_ind = np.argmax(scores)
        best_feature = features_used_list[best_feature_ind]
        best_score = scores[best_feature_ind]
        best_dataset = tensors[best_feature_ind]
        return best_score, best_feature, best_dataset
    

    def add_l_remove_r_one_iteration(self, existing_dataset):
        
        feature_set_to_add = self.all_feature_set - self.features_used
        print("feature_set_to_add", feature_set_to_add)

        # add l features
        for i in range(self.l):
            best_score, best_feature, best_dataset = self.find_feature_to_add(existing_dataset, feature_set_to_add)
            print(f"best_score: {best_score}, best_feature: {best_feature}")

            existing_dataset = best_dataset
            feature_set_to_add.remove(best_feature)
            self.features_used.add(best_feature)

        # remove r features
        for i in range(self.r):
            best_score, best_feature, best_dataset = self.find_feature_to_remove(existing_dataset)
            print(f"best_score: {best_score}, best_feature: {best_feature}")

            existing_dataset = best_dataset
            self.features_used.remove(best_feature)

        return best_score, existing_dataset
    
    def get_plus_l_minus_r(self):
        output_dict = {}

        if self.l > self.r:
            new_dataset = torch.zeros(self.dataset.shape) # initialize

            while len(self.features_used) <= self.num_features - self.l:
                best_score, new_dataset = self.add_l_remove_r_one_iteration(new_dataset)
                print(best_score)
                output_dict[",".join(str(x) for x in self.features_used)] = best_score

        else:
            # remove r
            # add m
            raise NotImplementedError
        
        # print output_dict
        for k, v in output_dict.items():
            print(f"{k}: {v}")

        return output_dict
    
    def evaluate(self):
        return self.get_plus_l_minus_r()


class SFS(FeatureSelectionAgorithm):
    """Implements SFS: Sequential Forward Selection. Based on the implementation here: http://enggjournals.com/ijcse/doc/IJCSE11-03-05-051.pdf"""
    def __init__(self, dataset, predict_score_method, l=1, features_dim=2, **prediction_args):
        super().__init__(dataset, predict_score_method, features_dim, **prediction_args)
        self.l = l


    def add_one_iteration(self, existing_dataset):
        feature_set_to_add = self.all_feature_set - self.features_used
        best_score, best_feature, best_dataset = self.find_feature_to_add(existing_dataset, feature_set_to_add)
        existing_dataset = best_dataset
        feature_set_to_add.remove(best_feature)
        self.features_used.add(best_feature)

        return best_score, existing_dataset


    def get_sfs(self):
        output_dict = {}

        # initialize
        new_dataset = torch.zeros(self.dataset.shape) # initialize
        while len(self.features_used) <= self.num_features - self.l - 1:
            best_score, new_dataset = self.add_one_iteration(new_dataset)
            output_dict[",".join(str(x) for x in self.features_used)] = best_score

        # print output_dict
        for k, v in output_dict.items():
            print(f"{k}: {v}")
        
        return output_dict

    def evaluate(self):
        return self.get_sfs()

