import argparse
from rnn_tasks import *
import pandas as pd

def inferInputInfo(fname, startId, targetStart):
    df = pd.read_table(fname)
    token_size = 0
    num_steps = 0
    update_token_size = True
    for i in range(startId, len(df.columns)):
        if df.columns[i].endswith('1'):
            update_token_size = False
        if update_token_size:
            token_size += 1
        if df.columns[i] == targetStart:
            break
        num_steps += 1
    return token_size, int(num_steps/token_size)

parser = argparse.ArgumentParser()
parser.add_argument('task', help='task type: train or score', type=str)
parser.add_argument('fname', help='path to input file', type=str)
parser.add_argument('targetStartName', help='column name of the first target variable', type=str)
parser.add_argument('--tokenSize', help='number of columns per time step', type=int)
parser.add_argument('--numSteps', help='number of time steps', type=int)
parser.add_argument('--startId', help='index of the first input column, default is 1', default=1, type=int)
parser.add_argument('--learningRate', help='initial learning rate before being divided by mini-batch size', default=0.5, type=float)
parser.add_argument('--batchSize', help='number of obs in a mini-batch', default=128, type=int)
parser.add_argument('--epochs', help='number of iterations for training', default=10, type=int)
parser.add_argument('--stepSize', help='number of steps before multiplying learning rate by gamma', default=10, type=int)
parser.add_argument('--gamma', help='gamma for adjusting learning rate', default=0.9, type=float)
parser.add_argument('--modelPath', help='path to model files', default='./model.ckpt', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    task = args.task
    fname = args.fname
    start_id = args.startId
    target_start = args.targetStartName
    learning_rate = args.learningRate
    batch_size = args.batchSize
    epochs = args.epochs
    step_size = args.stepSize
    gamma = args.gamma
    model_path = args.modelPath
    if args.tokenSize and args.numSteps:
        token_size = args.tokenSize
        num_steps = args.numSteps
    else:
        token_size, num_steps = inferInputInfo(fname, start_id, target_start)
    print('token size: %d\t number of steps: %d' % (token_size, num_steps))
    inputs, targets = getNumDataFromFile(fname, token_size*num_steps, target_start, 1, inputStartId=start_id)
    targetBins = [-0.01, 0, 0.01]
    discretizeTargets(targets, targetBins)
    if task.lower() == 'train':
        trainRnn(inputs, targets, None,
                 lr=learning_rate, epochs=epochs, rnnType='uni', task='perseq', stackedDimList=['1024'], cell='gru',
                 miniBatchSize=batch_size, tokenSize=token_size, nclass=len(targetBins)+1, seed=32145, gamma=gamma,
                 step_size=step_size, ckpt=model_path)
    elif task.lower() == 'score':
        scoreRnn(model_path, inputs, labels=targets)
    else:
        print('unrecognized task type: %s' % task)
