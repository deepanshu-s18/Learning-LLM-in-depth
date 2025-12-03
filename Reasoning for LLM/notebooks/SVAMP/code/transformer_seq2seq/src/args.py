import argparse

### Add Early Stopping ###

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Single sequence model')

	# Mode specifications
	parser.add_argument('-mode', type=str, default='train', choices=['train', 'test'], help='Modes: train, test')
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
	parser.set_defaults(histogram=False)
	parser.add_argument('-save_writer', dest='save_writer',action='store_true', help='To write tensorboard')
	parser.add_argument('-no-save_writer', dest='save_writer', action='store_false', help='Dont write tensorboard')
	parser.set_defaults(save_writer=False)

	# Device Configuration
	parser.add_argument('-gpu', type=int, default=2, help='Specify the gpu to use')
	parser.add_argument('-early_stopping', type=int, default=500, help='Early Stopping after n epoch')
	parser.add_argument('-seed', type=int, default=6174, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	parser.add_argument('-save_model', dest='save_model',action='store_true', help='To save the model')
	parser.add_argument('-no-save_model', dest='save_model', action='store_false', help='Dont save the model')
	parser.set_defaults(save_model=False)

	# Transformer parameters
	parser.add_argument('-heads', type=int, default=8, help='Number of Attention Heads')
	parser.add_argument('-encoder_layers', type=int, default=6, help='Number of layers in encoder')
	parser.add_argument('-decoder_layers', type=int, default=6, help='Number of layers in decoder')
	parser.add_argument('-d_model', type=int, default=300, help='the number of expected features in the encoder inputs') #768? features of BERT? HAS TO BE 300 if using word2Vec
