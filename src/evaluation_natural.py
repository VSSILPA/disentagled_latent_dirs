from torch.utils.data import DataLoader
import numpy as np
import random
import torch
import os
from src.models.closedform.utils import load_generator
from utils import NoiseDataset
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn.functional as F
from logger import PerfomanceLogger
import seaborn as sns

from models.attribute_predictors import attribute_predictor ,attribute_utils

sns.set_theme()
perf_logger = PerfomanceLogger()


def _set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class Evaluator():
    def __init__(self, random_seed, result_path, pretrained_path, num_directions, num_samples, z_batch_size, epsilon):
        _set_seed(random_seed)
        self.result_path = result_path
        self.pretrained_path = pretrained_path
        self.num_directions = num_directions
        self.directions_idx = list(range(0, self.num_directions)) ##TODOD change from 0 to 512
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.z_batch_size = z_batch_size
        self.num_batches = int(self.num_samples / self.z_batch_size)
        self.all_attr_list = ['pose','Receding_Hairline', 'Oval_Face', 'Bags_Under_Eyes', 'Smiling', 'Mouth_Slightly_Open',
                           'Bald', 'Bushy_Eyebrows', 'High_Cheekbones', 'Black_Hair', 'Heavy_Makeup', 'Pointy_Nose',
                           'Sideburns', 'Wearing_Lipstick', 'Chubby', 'Pale_Skin', 'Arched_Eyebrows', 'Big_Nose',
                           'No_Beard', 'Eyeglasses', 'Wearing_Earrings', 'Brown_Hair', 'Wearing_Hat', 'Goatee',
                           'Mustache', 'Narrow_Eyes', 'Double_Chin', 'Attractive', 'Young', 'Gray_Hair',
                           '5_o_Clock_Shadow', 'Big_Lips', 'Rosy_Cheeks', 'Wearing_Necktie', 'Male', 'Blurry',
                           'Wavy_Hair', 'Blond_Hair', 'Straight_Hair', 'Wearing_Necklace', 'Bangs']

        # ['pose', 'eyeglasses', 'male', 'smiling', 'young']
       # self.all_attr_list = ['Pose', 'Eyeglasses', 'Male', 'Smiling', 'Young']
        attr_index = list(range(len(self.all_attr_list)))
        self.attr_list_dict = OrderedDict(zip(self.all_attr_list, attr_index))

    def _get_predictor_list(self, attr_list , source ='nvidia'):
        predictor_list = []
        if source == 'nvidia':
            predictor = attribute_predictor.get_classifier(
                os.path.join(self.pretrained_path, "classifiers", 'pose', "weight.pkl"),
                'cuda')
            predictor.cuda().eval()
            predictor_list.append(predictor)
            for classifier_name in self.all_attr_list[1:]:
                predictor = attribute_utils.ClassifierWrapper(classifier_name, device='cuda')
                predictor.cuda().eval()
                predictor_list.append(predictor)
        else:
            for each in attr_list:
                predictor = attribute_predictor.get_classifier(
                    os.path.join(self.pretrained_path, "classifiers", each, "weight.pkl"),
                    'cuda')
                predictor.cuda().eval()
                predictor_list.append(predictor)
        return predictor_list

    def get_reference_attribute_scores(self, generator, z_loader, attribute_list):
        predictor_list = self._get_predictor_list(attribute_list)
        attribute_scores_ref = []
        with torch.no_grad():
            for z in z_loader:
                z = z.cuda()
                images = generator(z)
                images = (images + 1) / 2
                predict_images = F.avg_pool2d(images, 4, 4)
                img_score = []
                for predictor in predictor_list:
                    softmax_out = predictor(predict_images)
                    each_img_score = softmax_out[:, 1].detach()
                    # each_img_score = torch.softmax(predictor(predict_images), dim=1)[:, 1].detach()
                    img_score.append(each_img_score)
                attribute_scores_ref.append(img_score)
        torch.save(attribute_scores_ref, os.path.join(self.result_path, 'reference_attribute_scores.pkl'))
        return attribute_scores_ref

    def get_evaluation_metric_values(self, generator, deformator, attribute_list, reference_attr_scores, z_loader,
                                     directions_idx):
        predictor_list = self._get_predictor_list(attribute_list)
        rescoring_matrix = []
        all_dir_attr_manipulation_acc = []
        with torch.no_grad():
            for dir in directions_idx:
                perf_logger.start_monitoring("Direction " + str(dir) + " completed")
                attr_variation = [0.0] * len(predictor_list)
                attr_manipulation_acc = [0] * len(predictor_list)
                for i, z in enumerate(z_loader):
#                    perf_logger.start_monitoring("Batch" + str(i) + " completed")
                    perf_logger.start_monitoring("image_prep" + str(i) + " completed")

                    w_shift = z.detach().cpu() + deformator[dir: dir + 1] * self.epsilon
                    w_shift = w_shift.cuda()
                    images_shifted = generator(w_shift)
                    images_shifted = (images_shifted + 1) / 2
                    predict_images = F.avg_pool2d(images_shifted, 4, 4)
                    perf_logger.stop_monitoring("image_prep" + str(i) + " completed")

                    for pred_idx, predictor in enumerate(predictor_list):
                        perf_logger.start_monitoring("forward_pass" + str(i) + " completed")

                        softmax_values = torch.softmax(predictor(predict_images), dim=1)
                        perf_logger.stop_monitoring("forward_pass" + str(i) + " completed")

                        img_shift_score = softmax_values[:, 1].detach()
                        predictions = torch.argmax(softmax_values, dim=-1).sum()
                        delta = img_shift_score.detach() - reference_attr_scores[i][pred_idx]
                        attr_variation[pred_idx] = attr_variation[pred_idx] + delta.mean()
                        attr_manipulation_acc[pred_idx] = attr_manipulation_acc[pred_idx] + predictions.item()
                    del predict_images
#                    perf_logger.stop_monitoring("Batch" + str(i) + " completed")

                all_attr_variation = []
                for var in attr_variation:
                    all_attr_variation.append(round(abs((var / self.num_batches).item()), 2)), 2
                rescoring_matrix.append(all_attr_variation)
                all_dir_attr_manipulation_acc.append(np.array(attr_manipulation_acc) / self.num_samples)
                perf_logger.stop_monitoring("Direction " + str(dir) + " completed")

                torch.save(rescoring_matrix, os.path.join(self.result_path, 'rescoring matrix.pkl'))
                torch.save(all_dir_attr_manipulation_acc,
                   os.path.join(self.result_path, 'attribute manipulation accuracy.pkl'))


        rescoring_matrix = np.array(rescoring_matrix)
        all_dir_attr_manipulation_acc = np.array(all_dir_attr_manipulation_acc)

        torch.save(rescoring_matrix, os.path.join(self.result_path, 'rescoring matrix.pkl'))
        torch.save(all_dir_attr_manipulation_acc,
                   os.path.join(self.result_path, 'attribute manipulation accuracy.pkl'))
        #self.get_heat_map(rescoring_matrix, directions_idx, attribute_list, self.result_path)
        return rescoring_matrix, all_dir_attr_manipulation_acc

    def get_partial_metrics(self, attributes, direction_idx, attr_vs_direction, rescoring_matrix,
                            attr_manipulation_acc):
        dir_attr_path = os.path.join(self.result_path, 'Direction_vs_Classifier_Metrics')
        os.makedirs(dir_attr_path, exist_ok=True)
        selected_attr = {cls_key: self.attr_list_dict[cls_key] for cls_key in attributes}
        attr_indices = list(selected_attr.values())
        temp_matrix = rescoring_matrix[direction_idx]
        partial_rescoring_matrix = temp_matrix[:, attr_indices]
        self.get_heat_map(partial_rescoring_matrix, direction_idx, attributes, dir_attr_path, classifier='partial')
        for cls, dir in attr_vs_direction.items():
            acc = attr_manipulation_acc[dir, selected_attr[cls]]
            attr_vs_direction[cls] = acc

        with open(os.path.join(dir_attr_path, 'Attribute_manipulation_accuracies.json'),
                  'w') as fp:
            json.dump(attr_vs_direction, fp)
        return partial_rescoring_matrix, attr_vs_direction

    def get_classifer_analysis(self, attributes, dir_idx, top_k, rescoring_matrix, attr_manipulation_acc):
        selected_attr = {cls_key: self.attr_list_dict[cls_key] for cls_key in attributes}
        for cls, cls_index in selected_attr.items():
            classifier_direction_dict = {}
            classifier_analysis_result_path = os.path.join(self.result_path, cls)
            os.makedirs(classifier_analysis_result_path, exist_ok=True)
            rescoring_matrix = torch.FloatTensor(rescoring_matrix)
            classifier_variance = rescoring_matrix[:, cls_index]
            best_direction_indices = torch.sort(classifier_variance, descending=True)[1][0:top_k]
            top_k_directions = np.array(dir_idx)[best_direction_indices]
            top_k_directions = getattr(top_k_directions, "tolist", lambda: top_k_directions)()
            top_k_direction_acc = attr_manipulation_acc[:, cls_index][best_direction_indices]
            top_k_direction_acc = getattr(top_k_direction_acc, "tolist", lambda: top_k_direction_acc)()
            classifier_rescoring_matrix = rescoring_matrix[best_direction_indices]
            classifier_direction_dict[cls] = {'top_directions': top_k_directions,
                                              'top_directions attr manipulation accuracy': top_k_direction_acc}

            torch.save(classifier_rescoring_matrix,
                       os.path.join(classifier_analysis_result_path, cls + '_rescoring_matrix.pkl'))

            self.get_heat_map(classifier_rescoring_matrix, top_k_directions, attributes,
                              classifier_analysis_result_path, classifier=cls)
            with open(os.path.join(classifier_analysis_result_path, 'Classifier_top_directions_details.json'),
                      'w') as fp:
                json.dump(classifier_direction_dict, fp)

    def get_heat_map(self, matrix, dir, attribute_list, path, classifier='full'):
        ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap='Blues')
        ax.xaxis.tick_top()
        plt.xticks(np.arange(len(attribute_list)) + 0.5, labels=attribute_list)
        plt.yticks(np.arange(len(dir)) + 0.5, labels=dir)
        plt.savefig(os.path.join(path, classifier + '_Rescoring_Analysis' + '.jpeg'))
        plt.close()

    def evaluate_directions(self, deformator):
        generator = load_generator(None, model_name='pggan_celebahq1024')
        codes = torch.randn(self.num_samples, generator.z_space_dim).cuda()
        codes = generator.layer0.pixel_norm(codes)
        z = NoiseDataset(latent_codes=codes, num_samples=self.num_samples, z_dim=generator.z_space_dim)
        # z = torch.load(os.path.join(self.pretrained_path, 'z_analysis.pkl'))
        torch.save(z, os.path.join(self.result_path, 'z_analysis.pkl'))
        z_loader = DataLoader(z, batch_size=self.z_batch_size, shuffle=False)
        perf_logger.start_monitoring("Reference attribute scores done")
        reference_attr_scores = self.get_reference_attribute_scores(generator, z_loader, self.all_attr_list)
        perf_logger.stop_monitoring("Reference attribute scores done")
        perf_logger.start_monitoring("Metrics done")
        full_rescoring_matrix, full_attr_manipulation_acc = self.get_evaluation_metric_values(generator, deformator,
                                                                                              self.all_attr_list,
                                                                                              reference_attr_scores,
                                                                                              z_loader,
                                                                                              self.directions_idx)
        perf_logger.stop_monitoring("Metrics done")
        classifiers_to_analyse = self.all_attr_list
        top_k = 3
        self.get_classifer_analysis(classifiers_to_analyse, self.directions_idx, top_k, full_rescoring_matrix,
                                    full_attr_manipulation_acc)


if __name__ == '__main__':
    random_seed = 1234
    result_path = 'results/closed_form_celeba-qualitative trial'
    deformator_path = 'deformators/ClosedForm/pggan_celebahq1024'
    classifier_path = '/home/ubuntu/src/disentagled_latent_dirs/pretrained_models/'
    os.makedirs(result_path, exist_ok=True)

    pretrained_models_path = '/home/ubuntu/src/disentagled_latent_dirs/pretrained_models'
    num_directions = 5
    num_samples = 1000
    z_batch_size = 4
    epsilon = 2
    layers, deformator, eigen_values = torch.load(
        os.path.join(pretrained_models_path, deformator_path, 'pggan_celebahq1024.pkl'),
        map_location='cpu')
    deformator = torch.FloatTensor(deformator)
    evaluator = Evaluator(random_seed, result_path, pretrained_models_path, num_directions, num_samples, z_batch_size,
                          epsilon)
    evaluator.evaluate_directions(deformator)

    # attributes = ['male', 'pose']
    # rescoring_matrix = torch.load(os.path.join(result_path, 'rescoring matrix.pkl'))
    # attr_manipulation_acc = torch.load(os.path.join(result_path, 'attribute manipulation accuracy.pkl'))
    # direction_idx = [1, 2]
    # attr_vs_direction = {'male': 1, 'pose': 2}
    # evaluator.get_partial_metrics(attributes, direction_idx, attr_vs_direction, rescoring_matrix,
    #                     attr_manipulation_acc)
