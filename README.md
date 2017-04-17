# stock_prices

download_data.py: downloads daily prices of stocks listed in symbols.txt
prepare_data.py: converts long daily price sequences into same-length short sequences using adjusted closing price; allows start and end dates to separate training, validation and testing data
rnn_tasks.py: TensorFlow based RNN, capable of stacked LSTM/GRU; utility functions 
tkdl_util.py: utility functions to write weights to files
