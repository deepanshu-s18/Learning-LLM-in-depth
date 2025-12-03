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
