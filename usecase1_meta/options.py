import argparse


class Options(object):

    def __init__(self):

        # Handle command line arguments
        self.parser = argparse.ArgumentParser(
            description='Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments.')
        self.parser.add_argument('--plain_model_dir', default='./checkpoints/plain_model.torch',
                                 help='Directory to plain transformer model')
        self.parser.add_argument('--z2n_model_dir', default='./checkpoints/z2n_model.torch',
                                 help='Directory to trained Zomm2Net model') 
        self.parser.add_argument('--save_model_dir', default='./checkpoints/trained.torch',
                                 help='Directory to save model') 
        self.parser.add_argument('--task', choices={"train", "eval_downstream_task", "eval_timing", \
                                                    "eval_zoom_in_factor"},
                                 default="train",
                                 help=("Running objective/task: train an imputation model from scratch,\n"
                                       "                          run trained model on downstream tasks,\n"))
        self.parser.add_argument('--compute_baselines', default='False',
                                 help='Train baselines from scratch. If false, use saved data.') 
        # Data preparation
        self.parser.add_argument('--window_size', type=int, default=1000,
                                 help='Number of time series data in one pass')
        self.parser.add_argument('--window_skip', type=int, default=500,
                                 help='Number of time series data to skip in data preprocessing')
        self.parser.add_argument('--zoom_in_factor', type=int, default=50,
                                 help='zoom-in factor')

        # Transformer parameter
        self.parser.add_argument('--feat_dim', type=int, default=2,
                                 help='Input feature dimension')
        self.parser.add_argument('--dim_output', type=int, default=1,
                                 help='Output dimension')
        self.parser.add_argument('--d_model', type=int, default=40,
                                 help='d_model in transformer')
        self.parser.add_argument('--n_heads', type=int, default=4,
                                 help='Number of heads in transformer')
        self.parser.add_argument('--num_layers', type=int, default=1,
                                 help='Number of layers in transformer')
        self.parser.add_argument('--dim_feedforward', type=int, default=20,
                                 help='dim_feedforward in transformer')
        self.parser.add_argument('--lr', type=float, default=1e-3,
                                 help='Learning rate in training')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5,
                                 help='Weight decay in training')
        self.parser.add_argument('--dropout', type=float, default=0.2,
                                 help='Dropout in training')
        self.parser.add_argument('--emd_weight', type=float, default=0.01,
                                 help='Weight of EMD loss')   
        self.parser.add_argument('--batch_size', type=int, default=16,
                                 help='Batch size')  
        self.parser.add_argument('--n_epochs', type=int, default=300,
                                 help='Maximum number of training epochs') 
        self.parser.add_argument('--n_iters', type=int, default=100,
                                 help='Maximum number of Lagrangian Iterations')               
        # Lagrangian parameter
        self.parser.add_argument('--mu_lagrange', type=float, default=0.00001,
                                 help='mu parameter in lagrangian algorithm')


    def parse(self):

        args = self.parser.parse_args()

        return args
