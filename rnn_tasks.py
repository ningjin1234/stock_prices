import pandas
import tensorflow as tf
import numpy as np
from datetime import datetime as dt
import unittest
from tkdl_util import *
from tensorflow.python.ops import array_ops

def getRnnCell(nNeurons, cell='rnn', nCells=1, act=tf.tanh):
    ret = []
    rnnCell = None
    for i in range(nCells):
        if cell == 'rnn':
            rnnCell = tf.nn.rnn_cell.BasicRNNCell(nNeurons, activation=act)
        elif cell == 'gru':
            rnnCell = tf.nn.rnn_cell.GRUCell(nNeurons, activation=act)
        elif cell == 'lstm':
            rnnCell = tf.nn.rnn_cell.LSTMCell(nNeurons, activation=act, use_peepholes=True, forget_bias=1.0) 
        ret.append(rnnCell)
    if nCells == 1:
        return ret[0]
    return ret

def scaleToList(v, l):
    if isinstance(v, list):
        if len(v) == l:
            return v
        elif len(v) > l:
            return v[:l]
        else:
            ret = []
            for i in range(l):
                if i < len(v):
                    ret.append(v[i])
                else:
                    ret.append(v[-1])
            return ret
    return [v for i in range(l)]

def getRnnLayers(stackedDimList, inputData, inputLens, cellTypes='rnn', acts=tf.tanh, rnnTypes='uni'):
    tmpInputs = inputData
    cellTypes = scaleToList(cellTypes, len(stackedDimList))
    acts = scaleToList(acts, len(stackedDimList))
    rnnTypes = scaleToList(rnnTypes, len(stackedDimList)) 
    last_states = None
    for i in range(len(stackedDimList)):
        n = stackedDimList[i]
        cellType = cellTypes[i]
        act = acts[i]
        rnnType = rnnTypes[i]
        with tf.variable_scope('layer%d'%i):
            if rnnType == 'bi':
                cells = getRnnCell(n, cell=cellType, nCells=2, act=act)
                fwRnnCell = cells[0]
                bwRnnCell = cells[1]
                tmpSeq, tmp_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwRnnCell, cell_bw=bwRnnCell, dtype=tf.float32, 
                                                                     sequence_length=inputLens, inputs=tmpInputs)
                tmpInputs = tf.concat(2, [tmpSeq[0], tmpSeq[1]])
            elif rnnType == 'uni':
                cell = getRnnCell(n, cell=cellType, nCells=1, act=act)
                tmpSeq, tmp_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, sequence_length=inputLens, inputs=tmpInputs)
                tmpInputs = tmpSeq
            elif rnnType == 'rev':
                cell = getRnnCell(n, cell=cellType, nCells=1, act=act)
                inputDataReversed = array_ops.reverse_sequence(input=tmpInputs, seq_lengths=inputLens, seq_dim=1, batch_dim=0)
                raw_outputs_r, last_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, sequence_length=inputLens, inputs=inputDataReversed)
                tmpSeq = array_ops.reverse_sequence(input=raw_outputs_r, seq_lengths=inputLens, seq_dim=1, batch_dim=0)
                tmpInputs = tmpSeq
            else:
                raise ValueError("unsupported rnn type: %s" % rnnType)
    return tmpInputs, last_states       

def getRnnTrainOps(maxNumSteps=10, initEmbeddings=None, tokenSize=1,
                        bias_trainable=True, learningRate=0.1, rnnType='normal', stackedDimList=[],
                        act=tf.tanh, task='perseq', cell='rnn', nclass=0, seed=None):
    tf.reset_default_graph()
    if seed is not None:
        tf.set_random_seed(seed)
    inputTokens = tf.placeholder(tf.int32, [None, maxNumSteps])
    inputLens = tf.placeholder(tf.int32, [None])
    obsWgts = tf.placeholder(tf.float32, [None])
    if task.lower() in ['perseq']:
        if nclass <= 1:
            targets = tf.placeholder(tf.float32, [None, 1])
        else:
            targets = tf.placeholder(tf.int32, [None])
    elif task.lower() in ['pertoken', 'perstep']:   # corresponds to same-length output type in tkdlu; all input seqs must have same length
        if nclass <= 1:
            targets = tf.placeholder(tf.float32, [None, maxNumSteps])
        else:
            targets = tf.placeholder(tf.int32, [None])
    else:
        raise ValueError("unsupported task type: %s" % task)        

    if initEmbeddings is not None:
        embedding = tf.Variable(initEmbeddings, name='inputEmbeddings', trainable=False, dtype=tf.float32)
        inputData = tf.nn.embedding_lookup(embedding, inputTokens)
    else:
        inputTokens = tf.placeholder(tf.float32, [None, maxNumSteps, tokenSize])
        inputData = inputTokens

    tf.add_to_collection('inputData', inputData)
    tf.add_to_collection('feed_dict', inputTokens)
    tf.add_to_collection('feed_dict', inputLens)
    tf.add_to_collection('feed_dict', targets)
    tf.add_to_collection('feed_dict', obsWgts)
    cellTypes = scaleToList(cell, len(stackedDimList))
    acts = scaleToList(act, len(stackedDimList))
    rnnTypes = scaleToList(rnnType, len(stackedDimList)) 
    raw_outputs, last_states = getRnnLayers(stackedDimList, inputData, inputLens, cellTypes=cellTypes, rnnTypes=rnnTypes, acts=acts)
    nNeurons = stackedDimList[-1] if rnnTypes[-1] != 'bi' else 2*stackedDimList[-1]
    # print('number of neurons: %d' % nNeurons)
    flattened_outputs = tf.reshape(raw_outputs, [-1, nNeurons])
    if task.lower() in ['perseq']:
        batchSize = tf.shape(inputLens)[0]
        if rnnTypes[-1].lower() == 'rev':
            index = tf.range(0, batchSize) * maxNumSteps
        else:
            index = tf.range(0, batchSize) * maxNumSteps + inputLens - 1
        outputs = tf.gather(flattened_outputs, index)
    else:
        outputs = flattened_outputs
        if nclass <= 1:
            targets = tf.reshape(targets, [-1, 1])
        else:
            targets = tf.reshape(targets, [-1])
    nclass = 1 if nclass <= 1 else nclass
    outputW = tf.get_variable("outputW", [nNeurons, nclass], dtype=tf.float32)
    outputB = tf.get_variable("outputB", [nclass], dtype=tf.float32)
    tf.add_to_collection('outputB', outputB)
    prediction = tf.add(tf.matmul(outputs, outputW), outputB)
    tf.add_to_collection('prediction', prediction)
    if task.lower() in ['perseq']:
        if nclass <= 1:
            loss = tf.reduce_sum(tf.multiply(obsWgts, tf.pow(prediction-targets, 2))/2)
        else:
            logits = tf.reshape(prediction, [-1, nclass])
            softmax = tf.nn.softmax(logits) # for debugging purpose
            tf.add_to_collection('prediction', softmax)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
            loss = tf.reduce_sum(tf.multiply(obsWgts, losses))
    elif task.lower() in ['pertoken', 'perstep']:
        if nclass <= 1:
            loss = tf.reduce_sum(tf.multiply(obsWgts, tf.pow(prediction-targets, 2)/2/maxNumSteps))
        else:
            logits = tf.reshape(prediction, [-1, nclass])
            softmax = tf.nn.softmax(logits) # for debugging purpose
            tf.add_to_collection('prediction', softmax)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
            loss = tf.reduce_sum(tf.multiply(obsWgts, losses/maxNumSteps))
    tf.add_to_collection('loss', loss)
    lr = tf.Variable(learningRate, trainable=False)
    tvars = tf.trainable_variables()
#     optimizer = tf.train.GradientDescentOptimizer(lr)
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss, var_list=tvars) # for debugging purpose
    learningStep = optimizer.minimize(loss, var_list=tvars)
    initAll = tf.global_variables_initializer()
    # last return is output to screen for debugging purpose
    return inputTokens, inputLens, targets, prediction, loss, initAll, learningStep, gradients, lr, obsWgts

def genTextParms(docs, embeddingFile):
    textParms = {}
    print('loading embedding file %s' % embeddingFile)
    token2Id, embeddingArray = readEmbeddingFile(embeddingFile)
    maxNumSteps = 0
    lens = []
    for doc in docs:
        idList = tokens2ids(doc, token2Id)
        lens.append(len(idList))
        if len(idList) > maxNumSteps:
            maxNumSteps = len(idList)
    inputIds = []
    for doc in docs:
        ids = tokens2ids(doc, token2Id, maxNumSteps=maxNumSteps)
        inputIds.append(ids)
    inputIds = np.asarray(inputIds, dtype=np.int32)
    lens = np.asarray(lens, dtype=np.int32)
    embeddingArray = np.asarray(embeddingArray, dtype=np.float32)
    textParms['ids'] = inputIds
    textParms['lens'] = lens
    textParms['emb'] = embeddingArray
    textParms['maxl'] = maxNumSteps
    return textParms

def parseTextParms(inputTextParms):
    return inputTextParms['ids'], inputTextParms['lens'], inputTextParms['emb'], inputTextParms['maxl']

def get_batch_feed_dict(inputTokens, inputLens, inputIds, lens, labels, targets, start, end, task, maxNumSteps, obsWgts, obsWeights):
    if task.lower() in ['perseq']:
        sub_targets = labels[start:end]
    else:
        sub_targets = labels[start*maxNumSteps:end*maxNumSteps]
    return {inputTokens:inputIds[start:end], inputLens:lens[start:end], targets:sub_targets, obsWgts:obsWeights[start:end]}

def get_full_loss(sess, loss, ndocs, nbatches, miniBatchSize, inputTokens, inputLens, inputIds, lens, labels, targets, task, maxNumSteps, obsWgts, obsWeights):
    loss_val = 0.0
    for j in range(nbatches):
        start = miniBatchSize*j
        end = miniBatchSize*(j+1) if (j < nbatches-1) else ndocs
        feed_dict = get_batch_feed_dict(inputTokens, inputLens, inputIds, lens, labels, targets, start, end, task, maxNumSteps, obsWgts, obsWeights) 
        batch_loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_val += batch_loss_val
    return loss_val/ndocs

def scoreRnn(ckpt, docs, labels=None, inputTextParms=None, miniBatchSize=-1, embeddingFile=None, obsWeights=None):
    maxNumSteps = 0
    ndocs = len(docs)
    if miniBatchSize < 0:
        miniBatchSize = ndocs
    nbatches = int(ndocs/miniBatchSize)
    if ndocs % miniBatchSize > 0:
        nbatches += 1
    lens = []
    with tf.Session() as sess:
        print('importing meta graph....')
        saver = tf.train.import_meta_graph(ckpt+'.meta')
        print('restoring from latest checkpoint...')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        inputData = tf.get_collection('inputData')[0]
        outputB = tf.get_collection('outputB')[0]
        tokenSize = inputData.get_shape().as_list()[2]
        nclass = outputB.get_shape().as_list()[0]
        if inputTextParms is not None:
            inputIds, lens, embeddingArray, maxNumSteps = parseTextParms(inputTextParms)   
        elif embeddingFile is not None:
            inputTextParms = genTextParms(docs, embeddingFile) 
            inputIds, lens, embeddingArray, maxNumSteps = parseTextParms(inputTextParms)
        else:
            lens = [int(len(doc)/tokenSize) for doc in docs]
            lens = np.asarray(lens, dtype=np.int32)
            maxNumSteps = max(lens)
            embeddingArray = None
            inputIds = np.asarray(docs, dtype=np.float32)
            inputIds = np.reshape(inputIds, (ndocs, maxNumSteps, tokenSize))
            labels = np.asarray(labels, dtype=np.float32)
            labels = np.reshape(labels, (-1, 1))
            if nclass>1:
                labels = np.asarray(labels, dtype=np.int32)
                labels = np.reshape(labels, (-1))
        if obsWeights is None:
            obsWeights = np.ones(lens.shape)
        prediction = tf.get_collection('prediction')[0] if nclass <= 1 else tf.get_collection('prediction')[1]
        loss = tf.get_collection('loss')[0]
        inputTokens = tf.get_collection('feed_dict')[0]
        inputLens = tf.get_collection('feed_dict')[1]
        targets = tf.get_collection('feed_dict')[2]
        obsWgts = tf.get_collection('feed_dict')[3]
        feed_dict = {inputTokens:inputIds, inputLens:lens, targets:labels, obsWgts:obsWeights}
        print(sess.run(prediction, feed_dict=feed_dict))
        print(sess.run(loss, feed_dict=feed_dict)/ndocs)

def trainRnn(docs, labels, embeddingFile, miniBatchSize=-1, initWeightFile=None, trainedWeightFile=None, lr=0.1, epochs=1,
             rnnType='normal', stackedDimList=[], task='perseq', cell='rnn', tokenSize=1, nclass=0, seed=None,
             inputTextParms=None, gamma=0.5, step_size=50, ckpt=None, obsWeights=None):
    assert len(docs) == len(labels)
    maxNumSteps = 0
    ndocs = len(docs)
    if miniBatchSize < 0:
        miniBatchSize = ndocs
    nbatches = int(ndocs/miniBatchSize)
    if ndocs % miniBatchSize > 0:
        nbatches += 1
    lens = []
    if inputTextParms is not None:
        inputIds, lens, embeddingArray, maxNumSteps = parseTextParms(inputTextParms)   
    elif embeddingFile is not None:
        inputTextParms = genTextParms(docs, embeddingFile) 
        inputIds, lens, embeddingArray, maxNumSteps = parseTextParms(inputTextParms)
    else:
        lens = [int(len(doc)/tokenSize) for doc in docs]
        lens = np.asarray(lens, dtype=np.int32)
        maxNumSteps = max(lens)
        embeddingArray = None
        inputIds = np.asarray(docs, dtype=np.float32)
        inputIds = np.reshape(inputIds, (ndocs, maxNumSteps, tokenSize))
        labels = np.asarray(labels, dtype=np.float32)
        labels = np.reshape(labels, (-1, 1))
        if nclass>1:
            labels = np.asarray(labels, dtype=np.int32)
            labels = np.reshape(labels, (-1))
    if obsWeights is None:
        obsWeights = np.ones(lens.shape)
    inputTokens, inputLens, targets, prediction, loss, initAll, learningStep, gradients, learningRate, obsWgts = getRnnTrainOps(maxNumSteps=maxNumSteps,
                                                                                                   seed=seed, initEmbeddings=embeddingArray,
                                                                                                   learningRate=lr/miniBatchSize, rnnType=rnnType,
                                                                                                   stackedDimList=stackedDimList, task=task,
                                                                                                   cell=cell, tokenSize=tokenSize, nclass=nclass)
    tv_dict = dict()
    for v in tf.trainable_variables():
        tv_dict[v.name] = v
    saver = tf.train.Saver(tv_dict)
    # for d in docs[:10]:
    #     print(d)
    # for l in labels[:10]:
    #     print(l)
    print('learning rate: %f' % lr)
    print('rnn type: %s' % rnnType)
    print('cell type: %s' % cell)
    print('task type: %s' % task)
    print('mini-batch size: %d' % miniBatchSize)
    with tf.Session() as sess:
        sess.run(initAll)
        if initWeightFile is not None:
            ws = sess.run(tf.trainable_variables())
            writeWeightsWithNames(ws, tf.trainable_variables(), stackedDimList, initWeightFile)
        feed_dict = {inputTokens:inputIds, inputLens:lens, targets:labels, obsWgts:obsWeights}
#         print('loss before training: %.14g' % (sess.run(loss, feed_dict=feed_dict)/ndocs))
        full_loss = get_full_loss(sess, loss, ndocs, nbatches, miniBatchSize, inputTokens, inputLens, inputIds, lens, 
                                  labels, targets, task, maxNumSteps, obsWgts, obsWeights)
        print('loss before training: %.7g\t%s' % (full_loss, str(dt.now())))
        # print(sess.run(debugInfo, feed_dict=feed_dict))
        for i in range(epochs):
            if i % step_size == 0 and i != 0:
                lr *= gamma
            for j in range(nbatches):
                start = miniBatchSize*j
                if j < nbatches - 1:
                    end = miniBatchSize * (j+1)
                    if task.lower() in ['perseq']:
                        subTargets = labels[start:end]
                    else:
                        subTargets = labels[start*maxNumSteps:end*maxNumSteps]
                else:
                    end = ndocs
                    if task.lower() in ['perseq']:
                        subTargets = labels[start:end]
                    else:
                        subTargets = labels[start*maxNumSteps:end*maxNumSteps]
                sess.run(learningRate.assign(lr/(end-start)))
                feed_dict = {inputTokens:inputIds[start:end], inputLens:lens[start:end], targets:subTargets, obsWgts:obsWeights[start:end]}
#                 print('\tbefore batch %d: %.14g' % (j, sess.run(loss, feed_dict=feed_dict)/(end-start)))
                sess.run(learningStep, feed_dict=feed_dict)
            feed_dict = {inputTokens:inputIds, inputLens:lens, targets:labels, obsWgts:obsWeights}
#             print('loss after %d epochs: %.14g' % (i+1, sess.run(loss, feed_dict=feed_dict)/ndocs))
            full_loss = get_full_loss(sess, loss, ndocs, nbatches, miniBatchSize, inputTokens, inputLens, inputIds, lens, 
                                      labels, targets, task, maxNumSteps, obsWgts, obsWeights)
            print('loss after %d epochs: %.7g\tlearning rate: %.7g\t%s' % (i+1, full_loss, sess.run(learningRate), str(dt.now())))
        if ckpt is not None:    
            save_path = saver.save(sess, ckpt)
            print('model saved to %s' % save_path)
        if trainedWeightFile is not None:
            ws = sess.run(tf.trainable_variables())
            writeWeightsWithNames(ws, tf.trainable_variables(), stackedDimList, trainedWeightFile)

def getTextDataFromFile(fname, key='key', text='text', target='target', delimiter='\t'):
    table = pandas.read_table(fname)
    docs = table[text].values
    targets = table[target].values
    tokenized = []
    for doc in docs:
        tokenized.append(doc.split())
    labels = []
    for t in targets:
        labels.append([t])
    return tokenized, labels

def getNumDataFromFile(fname, inputLen, targetStartName, targetLen, delimiter='\t', inputStartId=1):
    inputs = []
    targets = []
    line_num = 0
    with open(fname, 'r') as fin:
        header = fin.readline().strip()
        header_dict = dict()
        for i,col in enumerate(header.split(delimiter)):
            header_dict[col] = i
        targetStartId = header_dict[targetStartName]
        for line in fin:
            line_num += 1
            splitted = line.strip().split(delimiter)
            invec = []
            outvec = []
            if len(splitted) < inputLen+targetLen:
                print(line_num, len(splitted), inputLen+targetLen)
            assert (len(splitted) >= inputLen+targetLen)
            for v in splitted[inputStartId:inputStartId+inputLen]:
                invec.append(float(v))
            for v in splitted[targetStartId:targetStartId+targetLen]:
                outvec.append(float(v))
            inputs.append(invec)
            targets.append(outvec)
    return inputs, targets

# this is needed to make TF and TKDLU use the same levelization
# there's no easy way to get the mapping, so currently I'm only testing binary targets for classification
def mapTargets(targets, targetMap):
    for arr in targets:
        for i in range(len(arr)):
            arr[i] = targetMap[arr[i]]

def discretizeTargets(targets, bins):
    tgtCnt = [0 for _ in range(len(bins)+1)]
    for arr in targets:
        for i in range(len(arr)):
            discretized = False
            for j in range(len(bins)):
                if arr[i] < bins[j]:
                    arr[i] = j
                    discretized = True
                    break
            if not discretized:
                arr[i] = len(bins)
            tgtCnt[arr[i]] += 1
    return tgtCnt

# assumes each obs has only one target value
def getObsWgtFromTgtCnt(targets, tgtCnt):
    suma = np.sum(tgtCnt)
    freq = [tgtCnt[i]/suma for i in range(len(tgtCnt))]
    res = []
    for arr in targets:
        res.append(1.0/freq[arr[0]])
    return res

if __name__ == '__main__':
    # NOTES:
    # 1. learning rate is divided by batch size during training, so with large batch size, learning rate is very small
    # 2. gamma-0.5 is too aggressive and doesn't work well
    # 3. currently it does NOT support training multiple targets of one sequence
    token_size = 35
    step_num = 4
    targetStartName = 'Adj Close SP5004'
    gamma = 0.9
    step_size = 20
    mini_batch = 128
    learning_rate = 0.5 
    epochs = 1
#     epochs = 1
    inputs, targets = getNumDataFromFile('index_training.txt', token_size*step_num, targetStartName, 1, inputStartId=1)
#     inputs, targets = getNumDataFromFile('index_test.txt', token_size*step_num, targetStartName, 1, inputStartId=1)
    print('number of obs: %d' % len(inputs))
    print('number of features per obs: %d' % len(inputs[0]))
    targetBins = [-0.01, 0.01]
    tgtCnt = discretizeTargets(targets, targetBins)
    weights = getObsWgtFromTgtCnt(targets, tgtCnt)
    trainRnn(inputs, targets, None,
             lr=learning_rate, epochs=epochs, rnnType='uni', task='perseq', stackedDimList=[1024], cell='gru', 
             miniBatchSize=mini_batch, tokenSize=token_size, nclass=len(targetBins)+1, seed=43215, gamma=gamma, 
             step_size=step_size, ckpt='./model.ckpt', obsWeights=weights)
#     scoreRnn('./model.ckpt', inputs, labels=targets)
    

