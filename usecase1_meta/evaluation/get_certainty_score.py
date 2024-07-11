
from model_training.utils import inference

def pred_multi(model_mc, i, WINDOW_SIZE, COARSE, T=10):
    model_mc.transformer_encoder.layers[0].dropout.train()
    model_mc.transformer_encoder.layers[0].dropout1.train()
    model_mc.transformer_encoder.layers[0].dropout2.train()
    model_mc.dropout1.train()
    model_mc.dropout2.train()
    model_mc.dropout3.train()
    res_dropout = []
    for _ in range(T):
        x = inference(model, test_dataset_10[i][0], WINDOW_SIZE=WINDOW_SIZE, COARSE=COARSE)[0][0].cpu().numpy()
        res_dropout.append(x)
    
    return res_dropout
