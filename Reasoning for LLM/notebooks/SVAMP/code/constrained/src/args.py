import argparse

### Add Early Stopping ###

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Single sequence model')

	# Mode specifications
	parser.add_argument('-mode', type=str, default='train', choices=['train', 'test', 'conf'], help='Modes: train, test, conf')
	parser.add_argument('-debug', dest='debug', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-debug', dest='debug', action='store_false', help='Operate in normal mode')
	parser.set_defaults(debug=False)

	# Run Config
	parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	parser.add_argument('-dataset', type=str, default='asdiv-a_fold0_final', help='Dataset')
	parser.add_argument('-display_freq', type=int, default= 10000, help='number of batches after which to display samples')
	parser.add_argument('-outputs', dest='outputs', action='store_true', help='Show full validation outputs')
	parser.add_argument('-no-outputs', dest='outputs', action='store_false', help='Do not show full validation outputs')
	parser.set_defaults(outputs=True)
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)

	# Meta Attributes
	parser.add_argument('-vocab_size', type=int, default=30000, help='Vocabulary size to consider')
	parser.add_argument('-histogram', dest='histogram', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-histogram', dest='histogram', action='store_false', help='Operate in normal mode')
	parser.set_defaults(histogram=True)
	parser.add_argument('-save_writer', dest='save_writer',action='store_true', help='To write tensorboard')
	parser.add_argument('-no-save_writer', dest='save_writer', action='store_false', help='Dont write tensorboard')
	parser.set_defaults(save_writer=False)

	# Device Configuration
	parser.add_argument('-gpu', type=int, default=1, help='Specify the gpu to use')
	parser.add_argument('-early_stopping', type=int, default=60, help='Early Stopping after n epoch')
	parser.add_argument('-seed', type=int, default=6174, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	parser.add_argument('-save_model', dest='save_model',action='store_true', help='To save the model')
	parser.add_argument('-no-save_model', dest='save_model', action='store_false', help='Dont save the model')
	parser.set_defaults(save_model=False)
	# parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')

	# LSTM parameters
	parser.add_argument('-emb2_size', type=int, default=16, help='Embedding dimensions of inputs')
	parser.add_argument('-cell_type', type=str, default='lstm', help='RNN cell for encoder and decoder, default: lstm')

	parser.add_argument('-use_attn', dest='use_attn',action='store_true', help='To use attention mechanism?')
	parser.add_argument('-no-attn', dest='use_attn', action='store_false', help='Not to use attention mechanism?')
	parser.set_defaults(use_attn=True)

	parser.add_argument('-attn_type', type=str, default='general', help='Attention mechanism: (general, concat), default: general')
	parser.add_argument('-hidden_size', type=int, default=384, help='Number of hidden units in each layer')
	parser.add_argument('-depth', type=int, default=2, help='Number of layers in each encoder and decoder')
	parser.add_argument('-dropout', type=float, default=0.1, help= 'Dropout probability for input/output/state units (0.0: no dropout)')
	parser.add_argument('-max_length', type=int, default=100, help='Specify max decode steps: Max length string to output')
	parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')
	parser.add_argument('-bidirectional', dest='bidirectional', action='store_true', help='Bidirectionality in LSTMs')
	parser.add_argument('-no-bidirectional', dest='bidirectional', action='store_false', help='Bidirectionality in LSTMs')
	parser.set_defaults(bidirectional=False)
