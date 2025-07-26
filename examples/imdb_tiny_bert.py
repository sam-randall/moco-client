

from datasets import load_dataset
from transformers import pipeline
from integrations.torch.bert_classification_model import EarlyExitTextClassificationModel
from src.ml_frame import MLFrame

def main():
    ds = load_dataset("stanfordnlp/imdb")
    train_ds = ds['train']
    test_ds = ds['test']

    tiny_bert = pipeline("text-classification", "arnabdhar/tinybert-imdb")

    tiny_bert.model.to('cpu')
    print("Loaded Tiny BERT")


    eetcm = EarlyExitTextClassificationModel(tiny_bert.model)
    
    frame = MLFrame.load_frame('../moco/imdb/out.frame')
    embedding = frame['bert.encoder.layer.0.attention']
    predictions = frame['predictions']
    eetcm.get_and_apply_rule(embedding, predictions)
    # N = embedding.shape[0]


if __name__ == "__main__":
    main()