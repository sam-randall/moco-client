import time
import torch
import json
import numpy as np
from typing import Literal
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os


def default_path(model):
    def compute(x):
        encoded = model.resnet.encoder(x)
        resnet_out = model.resnet.pooler(encoded[0])
        return model.classifier(resnet_out)
    return compute

def generate_embeddings(processor, model, dataset, from_layer: Literal["embedder"]):
    resnet_model = model.resnet
    if from_layer == 'embedder':
        decision_embedding = resnet_model.embedder

        embedding = []
        predictions = []
        for x in tqdm(dataset):
            x = x['image']
            try:
                x = processor(x, return_tensors = 'pt')
                with torch.no_grad():
                    e = decision_embedding(**x)
                    logits = model(**x).logits
                embedding.append(e.cpu().numpy())
                predictions.append(logits.cpu().numpy())
            except Exception as e:
                print(e)
        print("Stacking...")
        print(len(embedding), embedding[0].shape)
        embedding = np.vstack(embedding)
        print("Embedding done")
        predictions = np.vstack(predictions)
        return embedding, predictions

def main():
    SPLIT = 'train'
    dataset = load_dataset("microsoft/cats_vs_dogs")

    processor = AutoImageProcessor.from_pretrained("tangocrazyguy/resnet-50-finetuned-cats_vs_dogs")
    model = AutoModelForImageClassification.from_pretrained("tangocrazyguy/resnet-50-finetuned-cats_vs_dogs")
    model.eval()
    import inspect
    print(inspect.getsource(model.forward))
    print(inspect.getsource(model.resnet.forward))
    # return
    saved_embeddings_path = 'resnet_cats_dogs_embeddings.npy'
    if os.path.exists(saved_embeddings_path):
        e = np.load(saved_embeddings_path)
        p = np.load('resnet_cats_dogs_predictions.npy')
        from src.early_exit_model import EarlyExitModel
        eem = EarlyExitModel(model)
        classifications = p.argmax(axis = 1)
        # eem.run_fast_rule_job(saved_embeddings_path, "resnet_cats_dogs_predictions.npy", 1e-7, "sam.randall5@gmail.com")

    # Generate Embeddings, predictions for every image.
    # e, p = generate_embeddings(processor, model, dataset[SPLIT], "embedder") 
    # Call API to generate rules.
    # np.save("resnet_cats_dogs_embeddings.npy", e)
    # np.save("resnet_cats_dogs_predictions.npy", p)
    # Once you've called this API and have the rules available -- run system with the rules.

    if os.path.exists('rules.json'):
        with open('rules.json') as f:
            d = json.load(f)
            d = json.loads(d)
            print(d)
            
            eem.apply_rules_from_json_string(d)
            eem.decision_function = model.resnet.embedder
            eem.default_path = default_path(model)            
            # TODO: What to put here.
            new_ps = []
            start = time.time()
            for i, image in tqdm(enumerate(dataset[SPLIT])):
                x = None
                try:
                    x = processor(image['image'], return_tensors = 'pt')
                    
                    
                except Exception as e:
                    print(e)
                if x is not None:
                    p = eem.resnet_predict(**x)
                    new_ps.append(p)

            end = time.time()
            print("experimental time", end - start)


            predictions = torch.stack(new_ps)
            baseline_predictions = []
            start = time.time()
            for i, image in tqdm(enumerate(dataset[SPLIT])):
                try:
                    
                    with torch.no_grad():
                        x = processor(image['image'], return_tensors = 'pt')
                        out = model(**x).logits
                    baseline_predictions.append(out.argmax(axis = 1).cpu())
                except Exception as e:
                    print(e)
                
            end = time.time()
            print("Baseline time", end - start)



            baseline_predictions = torch.stack(baseline_predictions)
            print((predictions == baseline_predictions).sum(), predictions.shape[0])
if __name__ == "__main__":
    main()
