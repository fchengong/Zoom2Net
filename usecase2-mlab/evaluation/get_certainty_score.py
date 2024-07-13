
from model_training.utils import inference

def pred_withconf(model_mc, test_dataset, i, WINDOW_SIZE, COARSE, T=10):
    model_mc.transformer_encoder.layers[0].dropout.train()
    model_mc.transformer_encoder.layers[0].dropout1.train()
    model_mc.transformer_encoder.layers[0].dropout2.train()
    model_mc.dropout1.train()
    res_dropout = []
    for _ in range(T):
        x = inference(model_mc, test_dataset[i][0], WINDOW_SIZE=WINDOW_SIZE, COARSE=COARSE)[0][0].cpu().numpy()
        res_dropout.append(x)
    
    return res_dropout
