import face_recognition
import torch
import numpy as np


def one_hot(dims, value, indx):
    vec = torch.zeros(dims)
    vec[indx] = value
    return vec


def postprocess(images, min_val=-1.0, max_val=1.0):
    assert isinstance(images, torch.Tensor)
    images = images.detach().cpu().numpy()
    images = (images - min_val) * 255 / (max_val - min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    images = images.squeeze(0)
    return images


def compute_face_identity_score(generator, deformator, opt, attribute_idx, epsilon=3, num_samples=1000):
    cosine_similarity_metric = []
    euclidean_distance_metric = []
    face_identity_scores = {}
    for attribute in attribute_idx:
        count = 0
        correct = 0
        processed = 0
        for sample_idx in range(num_samples):
            z = torch.randn(opt.algo.eval.batch_size, opt.algo.ours.latent_dim).cuda()
            shift_epsilon = deformator(one_hot(512, epsilon, attribute).cuda()).view(1,-1)
            source_image = generator(z)
            shifted_image = generator(z + shift_epsilon)
            source_image = postprocess(source_image)
            shifted_image = postprocess(shifted_image)
            # source_image = (source_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1)/2
            try:
                source_image_encoding = face_recognition.face_encodings(source_image, model='large')[0]
                shifted_image_encoding = face_recognition.face_encodings(shifted_image, model='large')[0]
            except IndexError:
                count = count + 1
                print("skipped")
                print(count)
                print("Processed")
                print(sample_idx - count)
                continue
            cosine_similarity = torch.nn.functional.cosine_similarity(
                torch.FloatTensor(source_image_encoding).view(opt.algo.eval.batch_size, -1),
                torch.FloatTensor(shifted_image_encoding).view(opt.algo.eval.batch_size, -1))
            euclidean_distance = torch.cdist(
                torch.FloatTensor(source_image_encoding).view(opt.algo.eval.batch_size, -1),
                torch.FloatTensor(shifted_image_encoding).view(opt.algo.eval.batch_size, -1), p=2)
            correct = correct + int(face_recognition.compare_faces([source_image_encoding], shifted_image_encoding)[0])
            processed = processed + 1
            if sample_idx % 500 == 0:
                print("Finished " + str(sample_idx) + " images")
            cosine_similarity_metric.append(cosine_similarity)
            euclidean_distance_metric.append(euclidean_distance)
        avg_distance = sum(euclidean_distance_metric) / len(euclidean_distance_metric)
        avg_cosine_similarity = sum(cosine_similarity_metric) / len(cosine_similarity_metric)
        face_identity_scores[str(attribute)] = {'avg_distance': avg_distance,
                                                'avg_cosine_similarity': avg_cosine_similarity,
                                                'accuracy': float(correct / processed)}
        print(face_identity_scores)

    return face_identity_scores
