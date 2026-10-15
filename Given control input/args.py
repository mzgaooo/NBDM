import argparse
import configparser

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description='NBDM')
parser.add_argument('--config', type=str, default='./config_file/PEMS04.conf', help='Path to dataset configuration file')
args, _ = parser.parse_known_args()

config = configparser.ConfigParser()
config.read(args.config)

# data related
parser.add_argument('--data', type=str, default=config['data']['data'],help='dataset type')
parser.add_argument('--root_path', type=str, default=config['data']['root_path'],help='root path of the data file')
parser.add_argument('--data_path', type=str, default=config['data']['data_path'],help='data file')
parser.add_argument('--checkpoints', type=str, default=config['data']['checkpoints'],help='location of model checkpoints')
parser.add_argument('--features', type=str, default=config['data']['features'],
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default=config['data']['target'],help='target feature in S or MS task')
parser.add_argument('--num_workers', type=int, default=int(config['data']['num_workers']),help='number of nodes')

# training related
parser.add_argument('--device', type=str, default=config['train']['device'],help='device to run')
parser.add_argument('--epochs', type=int, default=int(config['train']['epochs']),help='train epochs')
parser.add_argument('--batch_size', type=int, default=int(config['train']['batch_size']),help='batch size')
parser.add_argument('--lr', type=float, default=float(config['train']['lr']),help='learning rate')
parser.add_argument('--lradj', type=str, default=(config['train']['lradj']),help='adjust learning rate')
parser.add_argument('--patience', type=int, default=int(config['train']['patience']),help='control the early stop')
parser.add_argument('--pct_start', type=float, default=float(config['train']['pct_start']),help='pct_start')
parser.add_argument('--use_amp', type=str_to_bool, default=str_to_bool(config['train']['use_amp']), help='use automatic mixed precision training')

# model related
parser.add_argument('--input_len', type=int, default=int(config['model']['input_len']), help='input sequence length')
parser.add_argument('--pred_len', type=int, default=int(config['model']['pred_len']), help='predict sequence length')

parser.add_argument('--in_dim', type=int, default=int(config['model']['in_dim']), help='inputs dimension')
parser.add_argument('--h_hid_dim', type=int, default=int(config['model']['h_hid_dim']), help='h hidden dimension')
parser.add_argument('--u_hid_dim', type=int, default=int(config['model']['u_hid_dim']), help='u hidden dimension')
parser.add_argument('--control_dim', type=int, default=int(config['model']['control_dim']), help='dim of nodes')

parser.add_argument('--num_nodes', type=int, default=int(config['model']['num_nodes']),help='number of nodes')
parser.add_argument('--dropout', type=int, default=float(config['model']['dropout']), help='dropout')
args = parser.parse_args()